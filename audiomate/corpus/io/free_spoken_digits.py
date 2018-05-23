import os
import glob

import audiomate
from audiomate.corpus import assets
from audiomate.utils import download
from audiomate.utils import files
from . import base

MASTER_DOWNLOAD_URL = 'https://github.com/Jakobovski/free-spoken-digit-dataset/archive/master.zip'


class FreeSpokenDigitDownloader(base.CorpusDownloader):
    """
    Downloader for the Free-Spoken-Digit dataset.

    Args:
        url (str): The url to download the dataset from. If not given the default URL is used.
                   It is expected to be a zip file.
    """

    def __init__(self, url=None):
        if url is None:
            self.url = MASTER_DOWNLOAD_URL
        else:
            self.url = url

    @classmethod
    def type(cls):
        return 'free-spoken-digits'

    def _download(self, target_path):
        os.makedirs(target_path, exist_ok=True)
        tmp_file = os.path.join(target_path, 'tmp_ark.zip')

        download.download_file(self.url, tmp_file)
        download.extract_zip(tmp_file, target_path)

        files.move_all_files_from_subfolders_to_top(target_path, delete_subfolders=True)

        os.remove(tmp_file)


class FreeSpokenDigitReader(base.CorpusReader):
    """
    Reader for the Free-Spoken-Digit Corpus.

    .. seealso::

       `Free-Spoken-Digit-Dataset <https://github.com/Jakobovski/free-spoken-digit-dataset>`_
          Download page
    """

    @classmethod
    def type(cls):
        return 'free-spoken-digits'

    def _check_for_missing_files(self, path):
        recordings_folder = os.path.join(path, 'recordings')

        if os.path.isdir(recordings_folder):
            return []
        else:
            return [recordings_folder]

    def _load(self, path):
        corpus = audiomate.Corpus(path=path)

        for file_path in glob.glob(os.path.join(path, 'recordings', '*.wav')):
            file_idx = os.path.splitext(os.path.basename(file_path))[0]

            corpus.new_file(file_path, file_idx)

            idx_parts = file_idx.split('_')
            digit = idx_parts[0]
            issuer_idx = '_'.join(idx_parts[1:-1])

            if issuer_idx not in corpus.issuers.keys():
                issuer = assets.Speaker(issuer_idx)
                corpus.import_issuers(issuer)

            utterance = corpus.new_utterance(file_idx, file_idx, issuer_idx)
            utterance.set_label_list(assets.LabelList(idx='transcription', labels=[
                assets.Label(str(digit))
            ]))

        return corpus
