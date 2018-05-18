import os
import collections
import zipfile
import shutil

import requests

import audiomate
from audiomate.corpus import assets
from audiomate.corpus import subset
from . import base
from audiomate.utils import textfile

DOWNLOAD_URL = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
META_FILE_PATH = os.path.join('meta', 'esc50.csv')


class ESC50Downloader(base.CorpusDownloader):
    """
    Downloader for the ESC-50 dataset.

    Args:
        url (str): The url to download the dataset from. If not given the default URL is used.
    """

    def __init__(self, url=None):
        if url is None:
            self.url = DOWNLOAD_URL
        else:
            self.url = url

    @classmethod
    def type(cls):
        return 'esc-50'

    def _download(self, target_path):
        temp_file = os.path.join(target_path, 'esc_50.zip')

        ESC50Downloader.download_file_chunked(self.url, temp_file)
        ESC50Downloader.extract_zip(temp_file, target_path)

        root_folder = os.path.join(target_path, 'ESC-50-master')

        for element in os.listdir(root_folder):
            shutil.move(os.path.join(root_folder, element), target_path)

        shutil.rmtree(root_folder)
        os.remove(temp_file)

    @staticmethod
    def download_file_chunked(url, to):
        """ Downloads the file from `url` to the local path `to`."""
        r = requests.get(url, stream=True)
        with open(to, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    @staticmethod
    def extract_zip(path, to):
        with zipfile.ZipFile(path) as archive:
            archive.extractall(to)


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
