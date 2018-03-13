import os
import collections

import pingu
from pingu.corpus import assets
from pingu.corpus import subset
from . import base
from pingu.utils import textfile

META_FILE_PATH = os.path.join('meta', 'esc50.csv')


class ESC50Reader(base.CorpusReader):
    """
    Reader for the ESC-50 dataset (Environmental Sound Classification).

    .. seealso::

       `ESC-50 <https://github.com/karoldvl/ESC-50>`_
          Download page
    """

    @classmethod
    def type(cls):
        return 'esc-50'

    def _check_for_missing_files(self, path):
        return []

    def _load(self, path):
        corpus = pingu.Corpus(path=path)

        meta_data = ESC50Reader.load_meta_data(path)

        folds = collections.defaultdict(list)
        esc10_utt_ids = []

        for record in meta_data:
            file_name = record[0]
            file_id = os.path.splitext(file_name)[0]
            file_path = os.path.abspath(os.path.join(path, 'audio', file_name))
            fold = record[1]
            category = record[3]
            esc10 = record[4]

            corpus.new_file(file_path, file_id)
            utt = corpus.new_utterance(file_id, file_id)
            utt.set_label_list(assets.LabelList(labels=[
                assets.Label(category)
            ]))

            folds['fold-{}'.format(fold)].append(file_id)

            if esc10 == 'True':
                esc10_utt_ids.append(file_id)

        for fold_id, fold_utt_ids in folds.items():
            fold_filter = subset.MatchingUtteranceIdxFilter(utterance_idxs=set(fold_utt_ids))
            fold_sv = subset.Subview(corpus, filter_criteria=[fold_filter])
            corpus.import_subview(fold_id, fold_sv)

        esc10_filter = subset.MatchingUtteranceIdxFilter(utterance_idxs=set(esc10_utt_ids))
        esc10_sv = subset.Subview(corpus, filter_criteria=[esc10_filter])
        corpus.import_subview('esc-10', esc10_sv)

        return corpus

    @staticmethod
    def load_meta_data(path):
        file_path = os.path.join(path, META_FILE_PATH)
        lines = textfile.read_separated_lines(file_path, separator=',')
        return lines[1:]
