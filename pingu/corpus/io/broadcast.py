import os

import pingu
from pingu.formats import audacity
from pingu.corpus import assets
from pingu.utils import textfile
from . import base

FILES_FILE_NAME = 'files.txt'
UTTERANCE_FILE_NAME = 'utterances.txt'
UTT_ISSUER_FILE_NAME = 'utt_issuers.txt'
LABEL_FILE = 'labels.txt'
FEAT_CONTAINER_FILE_NAME = 'features.txt'


class BroadcastLoader(base.CorpusLoader):
    """
    This is the corpus loader which is used for corpora where a separate label file per utterance exists.
    This especially is useful for corpora where the utterances are very long (e.g. broadcast recordings).
    """

    @classmethod
    def type(cls):
        return 'broadcast'

    def _check_for_missing_files(self, path):
        necessary_files = [FILES_FILE_NAME, UTTERANCE_FILE_NAME]
        missing_files = []

        for file_name in necessary_files:
            file_path = os.path.join(path, file_name)

            if not os.path.isfile(file_path):
                missing_files.append(file_name)

        return missing_files or None

    def _load(self, path):
        corpus = pingu.Corpus(path=path, loader=self)

        # Read files
        file_path = os.path.join(path, FILES_FILE_NAME)
        for file_idx, file_path in textfile.read_key_value_lines(file_path, separator=' ').items():
            corpus.new_file(os.path.join(path, file_path), file_idx=file_idx, copy_file=False)

        # Read utt to issuer mapping
        utt_issuer_path = os.path.join(path, UTT_ISSUER_FILE_NAME)
        utt_issuers = {}

        if os.path.isfile(utt_issuer_path):
            utt_issuers = textfile.read_key_value_lines(utt_issuer_path, separator=' ')

            for utterance_idx, issuer_idx in utt_issuers.items():
                if issuer_idx not in corpus.issuers.keys():
                    corpus.new_issuer(issuer_idx=issuer_idx)

        # Read utterances
        utterance_path = os.path.join(path, UTTERANCE_FILE_NAME)
        for utterance_idx, utt_info in textfile.read_separated_lines_with_first_key(utterance_path, separator=' ', max_columns=4).items():
            issuer_idx = None
            start = None
            end = None

            if len(utt_info) > 1:
                start = float(utt_info[1])

            if len(utt_info) > 2:
                end = float(utt_info[2])

            if utterance_idx in utt_issuers.keys():
                issuer_idx = utt_issuers[utterance_idx]

            corpus.new_utterance(utterance_idx, utt_info[0], issuer_idx=issuer_idx, start=start, end=end)

        # Read labels
        label_reference_file = os.path.join(path, LABEL_FILE)
        label_references = textfile.read_separated_lines(label_reference_file, separator=' ', max_columns=3)

        for record in label_references:
            utt_idx = record[0]
            label_path = os.path.join(path, record[1])
            label_idx = None

            if len(record) > 2:
                label_idx = record[2]

            entries = audacity.read_label_file(label_path)
            labels = [assets.Label(x[2], x[0], x[1]) for x in entries]

            corpus.new_label_list(utt_idx, idx=label_idx, labels=labels)

        # Read features
        feat_path = os.path.join(path, FEAT_CONTAINER_FILE_NAME)

        if os.path.isfile(feat_path):
            for container_name, container_path in textfile.read_key_value_lines(feat_path, separator=' ').items():
                corpus.new_feature_container(container_name, path=os.path.join(path, container_path))

        return corpus

    def _save(self, corpus, path):
        raise NotImplementedError("There is no implementation for saving in broadcast format.")
