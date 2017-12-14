import collections
import glob
import os

import pingu
from pingu.corpus import assets
from pingu.utils import textfile
from . import base

FILES_FILE_NAME = 'files.txt'
UTTERANCE_FILE_NAME = 'utterances.txt'
UTT_ISSUER_FILE_NAME = 'utt_issuers.txt'
LABEL_FILE_PREFIX = 'labels'
FEAT_CONTAINER_FILE_NAME = 'features.txt'


class DefaultReader(base.CorpusReader):
    """
    Reads corpora in the Default format.
    """

    @classmethod
    def type(cls):
        return 'default'

    def _check_for_missing_files(self, path):
        necessary_files = [FILES_FILE_NAME, UTTERANCE_FILE_NAME]
        missing_files = []

        for file_name in necessary_files:
            file_path = os.path.join(path, file_name)

            if not os.path.isfile(file_path):
                missing_files.append(file_name)

        return missing_files or None

    def _load(self, path):
        corpus = pingu.Corpus(path=path)

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
        utterances = textfile.read_separated_lines_with_first_key(utterance_path, separator=' ',
                                                                  max_columns=4)
        for utterance_idx, utt_info in utterances.items():
            issuer_idx = None
            start = 0
            end = -1

            if len(utt_info) > 1:
                start = float(utt_info[1])

            if len(utt_info) > 2:
                end = float(utt_info[2])

            if utterance_idx in utt_issuers.keys():
                issuer_idx = utt_issuers[utterance_idx]

            corpus.new_utterance(utterance_idx, utt_info[0], issuer_idx=issuer_idx, start=start,
                                 end=end)

        # Read labels
        for label_file in glob.glob(os.path.join(path, '{}_*.txt'.format(LABEL_FILE_PREFIX))):
            file_name = os.path.basename(label_file)
            key = file_name[len('{}_'.format(LABEL_FILE_PREFIX)):len(file_name) - len('.txt')]

            utterance_labels = collections.defaultdict(list)

            labels = textfile.read_separated_lines_generator(label_file, separator=' ',
                                                             max_columns=4)
            for record in labels:
                label = record[3]
                start = float(record[1])
                end = float(record[2])
                utterance_labels[record[0]].append(assets.Label(label, start, end))

            for utterance_idx, labels in utterance_labels.items():
                corpus.new_label_list(utterance_idx, key, labels)

        # Read features
        feat_path = os.path.join(path, FEAT_CONTAINER_FILE_NAME)

        if os.path.isfile(feat_path):
            containers = textfile.read_key_value_lines(feat_path, separator=' ')
            for container_name, container_path in containers.items():
                corpus.new_feature_container(container_name,
                                             path=os.path.join(path, container_path))

        return corpus


class DefaultWriter(base.CorpusWriter):
    """
    Writes corpora in the Default format.
    """

    @classmethod
    def type(cls):
        return 'default'

    def _save(self, corpus, path):
        # Write files
        file_path = os.path.join(path, FILES_FILE_NAME)
        file_records = [[file.idx, os.path.relpath(file.path, corpus.path)] for file in
                        corpus.files.values()]
        textfile.write_separated_lines(file_path, file_records, separator=' ', sort_by_column=0)

        # Write utterances
        utterance_path = os.path.join(path, UTTERANCE_FILE_NAME)
        utterance_records = {utterance.idx: [utterance.file_idx, utterance.start, utterance.end] for
                             utterance in corpus.utterances.values()}
        textfile.write_separated_lines(utterance_path, utterance_records, separator=' ',
                                       sort_by_column=0)

        # Write utt_issuers
        utt_issuer_path = os.path.join(path, UTT_ISSUER_FILE_NAME)
        utt_issuer_records = {utterance.idx: utterance.issuer_idx for utterance in
                              corpus.utterances.values()}
        textfile.write_separated_lines(utt_issuer_path, utt_issuer_records, separator=' ',
                                       sort_by_column=0)

        # Write labels
        for label_list_idx, label_lists in corpus.label_lists.items():
            file_path = os.path.join(path, '{}_{}.txt'.format(LABEL_FILE_PREFIX, label_list_idx))
            records = []

            for utterance_idx in sorted(label_lists.keys()):
                label_list = label_lists[utterance_idx]
                records.extend([(utterance_idx, l.start, l.end, l.value) for l in label_list])

            textfile.write_separated_lines(file_path, records, separator=' ')
