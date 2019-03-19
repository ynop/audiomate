import os

import pytest
import requests_mock

from audiomate import corpus
from audiomate.corpus import io
from audiomate.corpus.io import gtzan

from tests import resources
from . import reader_test as rt


@pytest.fixture()
def tar_data():
    with open(resources.get_resource_path(['sample_files', 'cv_corpus_v1.tar.gz']), 'rb') as f:
        return f.read()


class TestGtzanDownloader:

    def test_download(self, tar_data, tmpdir):
        target_folder = tmpdir.strpath
        downloader = io.GtzanDownloader()

        with requests_mock.Mocker() as mock:
            mock.get(gtzan.DOWNLOAD_URL, content=tar_data)

            downloader.download(target_folder)

        assert os.path.isfile(os.path.join(target_folder, 'cv-valid-dev.csv'))
        assert os.path.isdir(os.path.join(target_folder, 'cv-valid-dev'))
        assert os.path.isfile(os.path.join(target_folder, 'cv-valid-train.csv'))
        assert os.path.isdir(os.path.join(target_folder, 'cv-valid-train'))


class TestGtzanReader(rt.CorpusReaderTest):

    SAMPLE_PATH = resources.sample_corpus_path('gtzan')
    FILE_TRACK_BASE_PATH = SAMPLE_PATH

    EXPECTED_NUMBER_OF_TRACKS = 4
    EXPECTED_TRACKS = [
        rt.ExpFileTrack('bagpipe', os.path.join('music_wav', 'bagpipe.wav')),
        rt.ExpFileTrack('ballad', os.path.join('music_wav', 'ballad.wav')),
        rt.ExpFileTrack('acomic', os.path.join('speech_wav', 'acomic.wav')),
        rt.ExpFileTrack('acomic2', os.path.join('speech_wav', 'acomic2.wav')),
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 0

    EXPECTED_NUMBER_OF_UTTERANCES = 4
    EXPECTED_UTTERANCES = [
        rt.ExpUtterance('bagpipe', 'bagpipe', None, 0, float('inf')),
        rt.ExpUtterance('ballad', 'ballad', None, 0, float('inf')),
        rt.ExpUtterance('acomic', 'acomic', None, 0, float('inf')),
        rt.ExpUtterance('acomic2', 'acomic2', None, 0, float('inf')),
    ]

    EXPECTED_LABEL_LISTS = {
        'ballad': [
            rt.ExpLabelList(corpus.LL_DOMAIN, 1),
        ],
        'bagpipe': [
            rt.ExpLabelList(corpus.LL_DOMAIN, 1),
        ],
        'acomic': [
            rt.ExpLabelList(corpus.LL_DOMAIN, 1),
        ],
        'acomic2': [
            rt.ExpLabelList(corpus.LL_DOMAIN, 1),
        ],
    }

    EXPECTED_LABELS = {
        'bagpipe': [
            rt.ExpLabel(corpus.LL_DOMAIN, corpus.LL_DOMAIN_MUSIC, 0, float('inf')),
        ],
        'ballad': [
            rt.ExpLabel(corpus.LL_DOMAIN, corpus.LL_DOMAIN_MUSIC, 0, float('inf')),
        ],
        'acomic': [
            rt.ExpLabel(corpus.LL_DOMAIN, corpus.LL_DOMAIN_SPEECH, 0, float('inf')),
        ],
        'acomic2': [
            rt.ExpLabel(corpus.LL_DOMAIN, corpus.LL_DOMAIN_SPEECH, 0, float('inf')),
        ],
    }

    EXPECTED_NUMBER_OF_SUBVIEWS = 0

    def load(self):
        return io.GtzanReader().load(self.SAMPLE_PATH)
