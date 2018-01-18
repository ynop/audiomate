import collections
import glob
import os

import pingu
from pingu.corpus import assets
from pingu.corpus import subview
from pingu.utils import textfile
from . import base

FILES_FILE_NAME = 'files.txt'
UTTERANCE_FILE_NAME = 'utterances.txt'
UTT_ISSUER_FILE_NAME = 'utt_issuers.txt'
LABEL_FILE_PREFIX = 'labels'
FEAT_CONTAINER_FILE_NAME = 'features.txt'
SUBVIEW_FILE_PREFIX = 'subview'


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

        return missing_files

    def _load(self, path):
        file_path = os.path.join(path, FILES_FILE_NAME)
        utt_issuer_path = os.path.join(path, UTT_ISSUER_FILE_NAME)
        utterance_path = os.path.join(path, UTTERANCE_FILE_NAME)
        feat_path = os.path.join(path, FEAT_CONTAINER_FILE_NAME)

        corpus = pingu.Corpus(path=path)

        DefaultReader.read_files(file_path, corpus)
        utt_id_to_issuer = DefaultReader.read_utt_to_issuer_mapping(utt_issuer_path, corpus)
        DefaultReader.read_utterances(utterance_path, corpus, utt_id_to_issuer)
        DefaultReader.read_labels(path, corpus)
        DefaultReader.read_feature_containers(feat_path, corpus)
        DefaultReader.read_subviews(path, corpus)

        return corpus

    @staticmethod
    def read_files(file_path, corpus):
        path = os.path.dirname(file_path)
        for file_idx, file_path in textfile.read_key_value_lines(file_path, separator=' ').items():
            corpus.new_file(os.path.join(path, file_path), file_idx=file_idx, copy_file=False)

    @staticmethod
    def read_utt_to_issuer_mapping(utt_issuer_path, corpus):
        utt_issuers = {}

        if os.path.isfile(utt_issuer_path):
            for utt_id, issuer_idx in textfile.read_key_value_lines(utt_issuer_path, separator=' ').items():
                if issuer_idx in corpus.issuers.keys():
                    utt_issuers[utt_id] = corpus.issuers[issuer_idx]
                else:
                    utt_issuers[utt_id] = corpus.new_issuer(issuer_idx=issuer_idx)

        return utt_issuers

    @staticmethod
    def read_utterances(utterance_path, corpus, utt_idx_to_issuer):
        utterances = textfile.read_separated_lines_with_first_key(utterance_path, separator=' ', max_columns=4)

        for utterance_idx, utt_info in utterances.items():
            issuer_idx = None
            start = 0
            end = -1

            if len(utt_info) > 1:
                start = float(utt_info[1])

            if len(utt_info) > 2:
                end = float(utt_info[2])

            if utterance_idx in utt_idx_to_issuer.keys():
                issuer_idx = utt_idx_to_issuer[utterance_idx].idx

            corpus.new_utterance(utterance_idx, utt_info[0], issuer_idx=issuer_idx, start=start, end=end)

    @staticmethod
    def read_labels(path, corpus):
        for label_file in glob.glob(os.path.join(path, '{}_*.txt'.format(LABEL_FILE_PREFIX))):
            file_name = os.path.basename(label_file)
            key = file_name[len('{}_'.format(LABEL_FILE_PREFIX)):len(file_name) - len('.txt')]

            utterance_labels = collections.defaultdict(list)

            labels = textfile.read_separated_lines_generator(label_file, separator=' ', max_columns=4)

            for record in labels:
                label = record[3]
                start = float(record[1])
                end = float(record[2])
                utterance_labels[record[0]].append(assets.Label(label, start, end))

            for utterance_idx, labels in utterance_labels.items():
                ll = assets.LabelList(idx=key, labels=labels)
                corpus.utterances[utterance_idx].set_label_list(ll)

    @staticmethod
    def read_feature_containers(feat_path, corpus):
        if os.path.isfile(feat_path):
            base_path = os.path.dirname(feat_path)
            containers = textfile.read_key_value_lines(feat_path, separator=' ')
            for container_name, container_path in containers.items():
                corpus.new_feature_container(container_name, path=os.path.join(base_path, container_path))

    @staticmethod
    def read_subviews(path, corpus):
        for sv_file in glob.glob(os.path.join(path, '{}_*.txt'.format(SUBVIEW_FILE_PREFIX))):
            file_name = os.path.basename(sv_file)
            key = file_name[len('{}_'.format(SUBVIEW_FILE_PREFIX)):len(file_name) - len('.txt')]

            with open(sv_file, 'r') as f:
                content = f.read().strip()

            sv = subview.Subview.parse(content)
            corpus.import_subview(key, sv)


class DefaultWriter(base.CorpusWriter):
    """
    Writes corpora in the Default format.
    """

    @classmethod
    def type(cls):
        return 'default'

    def _save(self, corpus, path):
        file_path = os.path.join(path, FILES_FILE_NAME)
        utterance_path = os.path.join(path, UTTERANCE_FILE_NAME)
        utt_issuer_path = os.path.join(path, UTT_ISSUER_FILE_NAME)
        container_path = os.path.join(path, FEAT_CONTAINER_FILE_NAME)

        DefaultWriter.write_files(file_path, corpus)
        DefaultWriter.write_utterances(utterance_path, corpus)
        DefaultWriter.write_utt_to_issuer_mapping(utt_issuer_path, corpus)
        DefaultWriter.write_labels(path, corpus)
        DefaultWriter.write_feature_containers(container_path, corpus)
        DefaultWriter.write_subviews(path, corpus)

    @staticmethod
    def write_files(file_path, corpus):
        file_records = [[file.idx, os.path.relpath(file.path, corpus.path)] for file in corpus.files.values()]
        textfile.write_separated_lines(file_path, file_records, separator=' ', sort_by_column=0)

    @staticmethod
    def write_utterances(utterance_path, corpus):
        utterance_records = {utterance.idx: [utterance.file.idx, utterance.start, utterance.end] for
                             utterance in corpus.utterances.values()}
        textfile.write_separated_lines(utterance_path, utterance_records, separator=' ', sort_by_column=0)

    @staticmethod
    def write_utt_to_issuer_mapping(utt_issuer_path, corpus):
        utt_issuer_records = {utterance.idx: utterance.issuer.idx for utterance in corpus.utterances.values()}
        textfile.write_separated_lines(utt_issuer_path, utt_issuer_records, separator=' ', sort_by_column=0)

    @staticmethod
    def write_labels(path, corpus):
        records = collections.defaultdict(list)

        for utterance in corpus.utterances.values():
            for label_list_idx, label_list in utterance.label_lists.items():
                utt_records = [(utterance.idx, l.start, l.end, l.value) for l in label_list]
                records[label_list_idx].extend(utt_records)

        for label_list_idx, label_list_records in records.items():
            file_path = os.path.join(path, '{}_{}.txt'.format(LABEL_FILE_PREFIX, label_list_idx))
            textfile.write_separated_lines(file_path, label_list_records, separator=' ')

    @staticmethod
    def write_feature_containers(container_path, corpus):
        feat_records = [(idx, container.path) for idx, container in corpus.feature_containers.items()]
        textfile.write_separated_lines(container_path, feat_records, separator=' ')

    @staticmethod
    def write_subviews(path, corpus):
        for name, sv in corpus.subviews.items():
            sv_path = os.path.join(path, '{}_{}.txt'.format(SUBVIEW_FILE_PREFIX, name))
            with open(sv_path, 'w') as f:
                f.write(sv.serialize())
