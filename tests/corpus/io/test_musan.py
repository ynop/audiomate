import os

from audiomate import corpus
from audiomate import issuers
from audiomate.corpus import io
from audiomate.corpus.io import musan

import pytest
import requests_mock

from tests import resources
from . import reader_test as rt


@pytest.fixture()
def tar_data():
    with open(resources.get_resource_path(['sample_files', 'cv_corpus_v1.tar.gz']), 'rb') as f:
        return f.read()


class TestMusanDownloader:

    def test_download(self, tar_data, tmpdir):
        target_folder = tmpdir.strpath
        downloader = io.MusanDownloader()

        with requests_mock.Mocker() as mock:
            mock.get(musan.DOWNLOAD_URL, content=tar_data)

            downloader.download(target_folder)

        assert os.path.isfile(os.path.join(target_folder, 'cv-valid-dev.csv'))
        assert os.path.isdir(os.path.join(target_folder, 'cv-valid-dev'))
        assert os.path.isfile(os.path.join(target_folder, 'cv-valid-train.csv'))
        assert os.path.isdir(os.path.join(target_folder, 'cv-valid-train'))


class TestMusanReader(rt.CorpusReaderTest):

    SAMPLE_PATH = resources.sample_corpus_path('musan')
    FILE_TRACK_BASE_PATH = SAMPLE_PATH

    EXPECTED_NUMBER_OF_TRACKS = 5
    EXPECTED_TRACKS = [
        rt.ExpFileTrack('music-fma-0000', os.path.join('music', 'fma', 'music-fma-0000.wav')),
        rt.ExpFileTrack('noise-free-sound-0000', os.path.join('noise', 'free-sound', 'noise-free-sound-0000.wav')),
        rt.ExpFileTrack('noise-free-sound-0001', os.path.join('noise', 'free-sound', 'noise-free-sound-0001.wav')),
        rt.ExpFileTrack('speech-librivox-0000', os.path.join('speech', 'librivox', 'speech-librivox-0000.wav')),
        rt.ExpFileTrack('speech-librivox-0001', os.path.join('speech', 'librivox', 'speech-librivox-0001.wav')),
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 3
    EXPECTED_ISSUERS = [
        rt.ExpSpeaker('speech-librivox-0000', 1, issuers.Gender.MALE, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('speech-librivox-0001', 1, issuers.Gender.FEMALE, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpArtist('Quiet_Music_for_Tiny_Robots', 1, 'Quiet_Music_for_Tiny_Robots'),
    ]

    EXPECTED_NUMBER_OF_UTTERANCES = 5
    EXPECTED_UTTERANCES = [
        rt.ExpUtterance('music-fma-0000', 'music-fma-0000', 'Quiet_Music_for_Tiny_Robots', 0, float('inf')),
        rt.ExpUtterance('noise-free-sound-0000', 'noise-free-sound-0000', None, 0, float('inf')),
        rt.ExpUtterance('noise-free-sound-0001', 'noise-free-sound-0001', None, 0, float('inf')),
        rt.ExpUtterance('speech-librivox-0000', 'speech-librivox-0000', 'speech-librivox-0000', 0, float('inf')),
        rt.ExpUtterance('speech-librivox-0001', 'speech-librivox-0001', 'speech-librivox-0001', 0, float('inf')),
    ]

    EXPECTED_LABEL_LISTS = {
        'music-fma-0000': [
            rt.ExpLabelList(corpus.LL_DOMAIN, 1),
        ],
        'noise-free-sound-0000': [
            rt.ExpLabelList(corpus.LL_DOMAIN, 1),
        ],
        'noise-free-sound-0001': [
            rt.ExpLabelList(corpus.LL_DOMAIN, 1),
        ],
    }

    EXPECTED_LABELS = {
        'music-fma-0000': [
            rt.ExpLabel(corpus.LL_DOMAIN, 'music', 0, float('inf')),
        ],
        'noise-free-sound-0000': [
            rt.ExpLabel(corpus.LL_DOMAIN, 'noise', 0, float('inf')),
        ],
        'noise-free-sound-0001': [
            rt.ExpLabel(corpus.LL_DOMAIN, 'noise', 0, float('inf')),
        ],
    }

    EXPECTED_NUMBER_OF_SUBVIEWS = 0

    def load(self):
        return io.MusanReader().load(self.SAMPLE_PATH)
