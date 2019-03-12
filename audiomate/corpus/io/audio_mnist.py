import os
import glob

import audiomate
from audiomate import annotations
from audiomate import issuers
from audiomate.utils import jsonfile

from . import base
from . import downloader

MASTER_DOWNLOAD_URL = 'https://github.com/soerenab/AudioMNIST/archive/master.zip'


class AudioMNISTDownloader(downloader.ArchiveDownloader):
    """
    Downloader for the audioMNIST dataset.

    Args:
        url (str): The url to download the dataset from. If not given the default URL is used.
                   It is expected to be a zip file.
    """

    def __init__(self, url=None):
        if url is None:
            url = MASTER_DOWNLOAD_URL

        super(AudioMNISTDownloader, self).__init__(
            url,
            move_files_up=True
        )

    @classmethod
    def type(cls):
        return 'audio-mnist'


class AudioMNISTReader(base.CorpusReader):
    """
    Reader for the audioMNIST Corpus.

    .. seealso::

       `AudioMNIST-Dataset <https://github.com/soerenab/AudioMNIST>`_
          Download page
    """

    @classmethod
    def type(cls):
        return 'audio-mnist'

    def _check_for_missing_files(self, path):
        missing_files = []

        recordings_folder = os.path.join(path, 'data')
        if not os.path.isdir(recordings_folder):
            missing_files.append(recordings_folder)

        meta_file = os.path.join(path, 'data', 'audioMNIST_meta.txt')
        if not os.path.isfile(meta_file):
            missing_files.append(meta_file)

        return missing_files

    def _load(self, path):
        corpus = audiomate.Corpus(path=path)
        data_path = os.path.join(path, 'data')
        meta_data = AudioMNISTReader.load_speaker_meta(path)

        for speaker_idx in os.listdir(data_path):
            speaker_path = os.path.join(data_path, speaker_idx)

            if os.path.isdir(speaker_path):

                for file_path in glob.glob(os.path.join(speaker_path, '*.wav')):
                    file_idx = os.path.splitext(os.path.basename(file_path))[0]

                    corpus.new_file(file_path, file_idx)

                    idx_parts = file_idx.split('_')
                    digit = idx_parts[0]

                    if speaker_idx not in corpus.issuers.keys():
                        issuer = issuers.Speaker(
                            speaker_idx,
                            gender=AudioMNISTReader.get_gender(meta_data, speaker_idx),
                            age_group=AudioMNISTReader.get_age_group(meta_data, speaker_idx)
                        )
                        corpus.import_issuers(issuer)

                    utterance = corpus.new_utterance(file_idx, file_idx, speaker_idx)
                    utterance.set_label_list(annotations.LabelList.create_single(
                        str(digit),
                        idx=audiomate.corpus.LL_WORD_TRANSCRIPT
                    ))

        return corpus

    @staticmethod
    def load_speaker_meta(corpus_path):
        meta_file = os.path.join(corpus_path, 'data', 'audioMNIST_meta.txt')
        return jsonfile.read_json_file(meta_file)

    @staticmethod
    def get_gender(meta_data, speaker_idx):
        gender_str = meta_data[speaker_idx]['gender']

        if gender_str == 'male':
            return issuers.Gender.MALE
        elif gender_str == 'female':
            return issuers.Gender.FEMALE
        else:
            return issuers.Gender.UNKNOWN

    @staticmethod
    def get_age_group(meta_data, speaker_idx):
        age_str = int(meta_data[speaker_idx]['age'])

        if age_str < 12:
            return issuers.AgeGroup.CHILD
        elif age_str < 18:
            return issuers.AgeGroup.YOUTH
        elif age_str < 65:
            return issuers.AgeGroup.ADULT
        else:
            return issuers.AgeGroup.SENIOR
