import os

import audiomate
from audiomate import annotations
from audiomate.utils import download
from audiomate.utils import files
from . import base

DOWNLOAD_URL = 'http://opihi.cs.uvic.ca/sound/music_speech.tar.gz'
DIRECTORIES = {'music_wav': 'music', 'speech_wav': 'speech'}


class GtzanDownloader(base.CorpusDownloader):
    """
    Downloader for the GTZAN Corpus.

    Args:
        url (str): The url to download the dataset from. If not given the default URL is used.
                   It is expected to be a tar.gz file.
    """

    def __init__(self, url=None):
        if url is None:
            self.url = DOWNLOAD_URL
        else:
            self.url = url

    @classmethod
    def type(cls):
        return 'gtzan'

    def _download(self, target_path):
        os.makedirs(target_path, exist_ok=True)
        tmp_file = os.path.join(target_path, 'tmp_ark.tar.gz')

        download.download_file(self.url, tmp_file)
        download.extract_tar(tmp_file, target_path)

        # We use copy since subfolders in the archive are read-only, hence throws permission error when trying to move.
        files.move_all_files_from_subfolders_to_top(target_path, delete_subfolders=True, copy=True)

        os.remove(tmp_file)


class GtzanReader(base.CorpusReader):
    """
    Reader for the GTZAN music/speech corpus. The corpus consits of 64 music and 64 speech tracks that are each 30
    seconds long. The Wave files are 16-bit mono and have a sampling rate of 22050 Hz.

    .. seealso::

       `MARSYAS: GTZAN Music/Speech <https://marsyasweb.appspot.com/download/data_sets/>`_
          Download page
    """

    @classmethod
    def type(cls):
        return 'gtzan'

    def _check_for_missing_files(self, path):
        return []

    def _load(self, path):
        corpus = audiomate.Corpus(path=path)

        for directory, type_name in DIRECTORIES.items():
            source_directory = os.path.join(path, directory)

            if not os.path.isdir(source_directory):
                continue

            it = os.scandir(source_directory)

            for entry in it:
                if not entry.name.endswith('.wav'):
                    continue

                file_path = os.path.join(source_directory, entry.name)
                file_idx = entry.name[0:-4]  # chop of .wav
                utterance_idx = file_idx  # every file is a separate utterance

                corpus.new_file(file_path, track_idx=file_idx, copy_file=False)
                utterance = corpus.new_utterance(utterance_idx, file_idx)
                utterance.set_label_list(annotations.LabelList.create_single(type_name, idx=audiomate.corpus.LL_DOMAIN))

        return corpus
