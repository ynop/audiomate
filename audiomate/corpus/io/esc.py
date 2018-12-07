import os
import collections

import audiomate
from audiomate import annotations
from audiomate.corpus import subset
from audiomate.utils import textfile
from . import base
from . import downloader

DOWNLOAD_URL = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
META_FILE_PATH = os.path.join('meta', 'esc50.csv')


class ESC50Downloader(downloader.ArchiveDownloader):
    """
    Downloader for the ESC-50 dataset.

    Args:
        url (str): The url to download the dataset from. If not given the default URL is used.
    """

    def __init__(self, url=None):
        if url is None:
            url = DOWNLOAD_URL

        super(ESC50Downloader, self).__init__(
            url,
            move_files_up=True
        )

    @classmethod
    def type(cls):
        return 'esc-50'


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
        corpus = audiomate.Corpus(path=path)

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
            utt.set_label_list(annotations.LabelList.create_single(category, idx=audiomate.corpus.LL_SOUND_CLASS))

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
