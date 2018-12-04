import os
import re

import audiomate
from audiomate import annotations
from audiomate.utils import download
from audiomate.utils import files
from . import base

DATA_URL = 'http://asi.insa-rouen.fr/enseignants/~arakoto/data_rouen.zip'

LABEL_PATTERN = r'(.*?)(\d+)'


class RouenDownloader(base.CorpusDownloader):
    """
    Downloader for the LITIS Rouen Audio scene dataset.
    """

    @classmethod
    def type(cls):
        return 'rouen'

    def _download(self, target_path):
        os.makedirs(target_path, exist_ok=True)
        tmp_file = os.path.join(target_path, 'tmp_ark.zip')

        download.download_file(DATA_URL, tmp_file)
        download.extract_zip(tmp_file, target_path)

        files.move_all_files_from_subfolders_to_top(target_path)

        os.remove(tmp_file)


class RouenReader(base.CorpusReader):
    """
    Reader for the LITIS Rouen Audio scene dataset.

    .. seealso::

       `Rouen <https://sites.google.com/site/alainrakotomamonjy/home/audio-scene>`_
          Download page
    """

    @classmethod
    def type(cls):
        return 'rouen'

    def _check_for_missing_files(self, path):
        return []

    def _load(self, path):
        corpus = audiomate.Corpus(path=path)
        regex = re.compile(LABEL_PATTERN)

        for file_name in os.listdir(path):
            base_name, ext = os.path.splitext(file_name)

            if ext == '.wav':
                file_path = os.path.join(path, file_name)

                match = regex.match(base_name)
                label = match.group(1)

                corpus.new_file(file_path, base_name)
                utt = corpus.new_utterance(base_name, base_name)
                ll = annotations.LabelList.create_single(
                    label,
                    idx=audiomate.corpus.LL_SOUND_CLASS
                )
                utt.set_label_list(ll)

        return corpus
