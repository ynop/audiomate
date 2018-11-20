import os
import collections

import audiomate
from audiomate import annotations
from audiomate.corpus import subset
from . import base
from audiomate.utils import textfile


class Urbansound8kReader(base.CorpusReader):
    """
    Reader for the Urbansound8k dataset.

    .. seealso::

       `Urbansound8k <http://urbansounddataset.weebly.com/urbansound8k.html>`_
          Download page
    """

    @classmethod
    def type(cls):
        return 'urbansound8k'

    def _check_for_missing_files(self, path):
        meta_file_path = os.path.join(path, 'metadata', 'UrbanSound8K.csv')

        if os.path.isfile(meta_file_path):
            return []
        else:
            return [meta_file_path]

    def _load(self, path):
        corpus = audiomate.Corpus(path=path)

        meta_file_path = os.path.join(path, 'metadata', 'UrbanSound8K.csv')
        meta_data = textfile.read_separated_lines(meta_file_path, separator=',', max_columns=8)[1:]

        folds = collections.defaultdict(set)

        for record in meta_data:
            file_name = record[0]
            fold = record[5]
            label = record[7]

            file_path = os.path.join(path, 'audio', 'fold{}'.format(fold), file_name)
            if os.path.isfile(file_path):
                basename = os.path.splitext(file_name)[0]

                corpus.new_file(file_path, basename)
                utt = corpus.new_utterance(basename, basename)
                utt.set_label_list(annotations.LabelList.create_single(label, idx=audiomate.corpus.LL_SOUND_CLASS))
                folds['fold{}'.format(fold)].add(basename)

        for fold_idx, fold_utterance_ids in folds.items():
            filter = subset.MatchingUtteranceIdxFilter(utterance_idxs=fold_utterance_ids)
            subview = subset.Subview(corpus, filter_criteria=[filter])

            corpus.import_subview(fold_idx, subview)

        return corpus
