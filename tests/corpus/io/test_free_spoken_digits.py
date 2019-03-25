import os

import pytest
import requests_mock

from audiomate import corpus
from audiomate import issuers
from audiomate.corpus.io import free_spoken_digits

from tests import resources
from . import reader_test as rt


@pytest.fixture()
def zip_data():
    path = resources.get_resource_path([
        'sample_files',
        'zip_sample_with_subfolder.zip'
    ])

    with open(path, 'rb') as f:
        return f.read()


class TestFreeSpokenDigitDownloader:

    def test_download(self, zip_data, tmpdir):
        target_folder = tmpdir.strpath
        downloader = free_spoken_digits.FreeSpokenDigitDownloader()

        with requests_mock.Mocker() as mock:
            mock.get(free_spoken_digits.MASTER_DOWNLOAD_URL, content=zip_data)

            downloader.download(target_folder)

        assert os.path.isfile(os.path.join(target_folder, 'a.txt'))
        assert os.path.isfile(os.path.join(target_folder, 'subsub', 'b.txt'))
        assert os.path.isfile(os.path.join(target_folder, 'subsub', 'c.txt'))


class TestFreeSpokenDigitReader(rt.CorpusReaderTest):

    SAMPLE_PATH = resources.sample_corpus_path('free_spoken_digits')
    FILE_TRACK_BASE_PATH = os.path.join(SAMPLE_PATH, 'recordings')

    EXPECTED_NUMBER_OF_TRACKS = 4
    EXPECTED_TRACKS = [
        rt.ExpFileTrack('0_jackson_0', '0_jackson_0.wav'),
        rt.ExpFileTrack('1_jackson_0', '1_jackson_0.wav'),
        rt.ExpFileTrack('2_theo_0', '2_theo_0.wav'),
        rt.ExpFileTrack('2_theo_1', '2_theo_1.wav'),
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 2
    EXPECTED_ISSUERS = [
        rt.ExpSpeaker('jackson', 2, issuers.Gender.UNKNOWN,
                      issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('theo', 2, issuers.Gender.UNKNOWN,
                      issuers.AgeGroup.UNKNOWN, None)
    ]

    EXPECTED_NUMBER_OF_UTTERANCES = 4
    EXPECTED_UTTERANCES = [
        rt.ExpUtterance('0_jackson_0', '0_jackson_0', 'jackson', 0, float('inf')),
        rt.ExpUtterance('1_jackson_0', '1_jackson_0', 'jackson', 0, float('inf')),
        rt.ExpUtterance('2_theo_0', '2_theo_0', 'theo', 0, float('inf')),
        rt.ExpUtterance('2_theo_1', '2_theo_1', 'theo', 0, float('inf'))
    ]

    EXPECTED_LABEL_LISTS = {
        '0_jackson_0': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
        ],
        '1_jackson_0': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
        ],
        '2_theo_0': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
        ],
        '2_theo_1': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1)
        ],
    }

    EXPECTED_LABELS = {
        '0_jackson_0': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, '0', 0, float('inf')),
        ],
        '1_jackson_0': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, '1', 0, float('inf')),
        ],
        '2_theo_0': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, '2', 0, float('inf')),
        ],
        '2_theo_1': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, '2', 0, float('inf')),
        ],
    }

    EXPECTED_NUMBER_OF_SUBVIEWS = 0

    def load(self):
        reader = free_spoken_digits.FreeSpokenDigitReader()
        return reader.load(self.SAMPLE_PATH)
