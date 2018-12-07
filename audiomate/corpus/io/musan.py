import os

import audiomate
from audiomate import annotations
from audiomate import issuers
from audiomate.utils import textfile
from . import base
from . import downloader

DOWNLOAD_URL = 'http://www.openslr.org/resources/17/musan.tar.gz'

AUDIO_TYPES_ = ['music', 'noise', 'speech']

ANN_NUM_COLUMS_ = {'music': 4, 'noise': 1, 'speech': 3}

ANN_FILE_NAME_ = 'ANNOTATIONS'


class MusanDownloader(downloader.ArchiveDownloader):
    """
    Downloader for the MUSAN Corpus.

    Args:
        url (str): The url to download the dataset from. If not given the default URL is used.
                   It is expected to be a tar.gz file.
    """

    def __init__(self, url=None):
        if url is None:
            url = DOWNLOAD_URL

        super(MusanDownloader, self).__init__(
            url,
            move_files_up=True
        )

    @classmethod
    def type(cls):
        return 'musan'


class MusanReader(base.CorpusReader):
    """
    Reader for the MUSAN corpus. MUSAN is a corpus of music, speech, and noise recordings.

    .. seealso::

       `MUSAN: A Music, Speech, and Noise Corpus <https://arxiv.org/pdf/1510.08484v1.pdf>`_
          Paper explaining the structure and characteristics of the corpus

       `OpenSLR: MUSAN <http://www.openslr.org/17/>`_
          Download page
    """

    @classmethod
    def type(cls):
        return 'musan'

    def _check_for_missing_files(self, path):
        # Some label files are missing anyway in the original data set
        # (e.g. speech/us-gov/ANNOTATIONS). What's left would be checking for missing directories.
        return []

    def _load(self, path):
        create_or_get_issuer = {
            'music': self._create_or_get_music_issuer,
            'noise': self._create_or_get_noise_issuer,
            'speech': self._create_or_get_speech_issuer,
        }

        corpus = audiomate.Corpus(path=path)

        for type_name, type_directory in self._directories(path).items():
            for _, source_directory in self._directories(type_directory).items():
                labels_path = os.path.join(source_directory, ANN_FILE_NAME_)
                labels = {}

                if os.path.exists(labels_path):
                    labels = textfile.read_separated_lines_with_first_key(
                        labels_path, separator=' ', max_columns=ANN_NUM_COLUMS_[type_name])

                it = os.scandir(source_directory)

                for entry in it:
                    if not entry.name.endswith('.wav'):
                        continue

                    file_path = os.path.join(source_directory, entry.name)
                    file_idx = entry.name[0:-4]  # chop of .wav
                    utterance_idx = file_idx  # every file is a separate utterance
                    issuer_idx = create_or_get_issuer[type_name](corpus, file_idx, labels)

                    corpus.new_file(file_path, track_idx=file_idx, copy_file=False)
                    utterance = corpus.new_utterance(utterance_idx, file_idx, issuer_idx)
                    utterance.set_label_list(annotations.LabelList.create_single(
                        type_name, idx=audiomate.corpus.LL_DOMAIN))

        return corpus

    @staticmethod
    def _directories(path):
        directories = {}
        it = os.scandir(path)
        for entry in it:
            if not entry.is_dir():
                continue

            directories[entry.name] = os.path.join(path, entry.name)

        return directories

    # noinspection PyUnusedLocal
    @staticmethod
    def _create_or_get_noise_issuer(corpus, file_idx, labels):
        return None

    @staticmethod
    def _create_or_get_music_issuer(corpus, file_idx, labels):
        if file_idx not in labels:
            return None

        issuer_idx = labels[file_idx][2]

        if issuer_idx not in corpus.issuers:
            issuer = issuers.Artist(issuer_idx, name=issuer_idx)
            corpus.import_issuers(issuer)

        return issuer_idx

    @staticmethod
    def _create_or_get_speech_issuer(corpus, file_idx, labels):
        if file_idx not in labels:
            return None

        issuer = issuers.Speaker(file_idx)

        if file_idx in labels:
            if labels[file_idx][0] == 'm':
                issuer.gender = issuers.Gender.MALE
            elif labels[file_idx][0] == 'f':
                issuer.gender = issuers.Gender.FEMALE

        corpus.import_issuers(issuer)

        return file_idx
