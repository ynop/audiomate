import os

import pytest
import requests_mock

from audiomate import corpus
from audiomate.corpus import io
from audiomate.corpus.io import rouen

from tests import resources
from . import reader_test as rt


@pytest.fixture
def downloader():
    return io.RouenDownloader()


@pytest.fixture()
def zip_data():
    with open(resources.get_resource_path(['sample_files', 'zip_sample_with_subfolder.zip']), 'rb') as f:
        return f.read()


class TestRouenDownloader:

    def test_download(self, zip_data, downloader, tmpdir):
        target_folder = tmpdir.strpath

        with requests_mock.Mocker() as mock:
            mock.get(rouen.DATA_URL, content=zip_data)

            downloader.download(target_folder)

        assert os.path.isfile(os.path.join(target_folder, 'a.txt'))
        assert os.path.isfile(os.path.join(target_folder, 'subsub', 'b.txt'))
        assert os.path.isfile(os.path.join(target_folder, 'subsub', 'c.txt'))


class TestRouenReader(rt.CorpusReaderTest):

    SAMPLE_PATH = resources.sample_corpus_path('rouen')
    FILE_TRACK_BASE_PATH = SAMPLE_PATH

    EXPECTED_NUMBER_OF_TRACKS = 4
    EXPECTED_TRACKS = [
        rt.ExpFileTrack('avion1', 'avion1.wav'),
        rt.ExpFileTrack('avion2', 'avion2.wav'),
        rt.ExpFileTrack('bus1', 'bus1.wav'),
        rt.ExpFileTrack('metro_rouen22', 'metro_rouen22.wav'),
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 0

    EXPECTED_NUMBER_OF_UTTERANCES = 4
    EXPECTED_UTTERANCES = [
        rt.ExpUtterance('avion1', 'avion1', None, 0, float('inf')),
        rt.ExpUtterance('avion2', 'avion2', None, 0, float('inf')),
        rt.ExpUtterance('bus1', 'bus1', None, 0, float('inf')),
        rt.ExpUtterance('metro_rouen22', 'metro_rouen22', None, 0, float('inf')),
    ]

    EXPECTED_LABEL_LISTS = {
        'avion1': [rt.ExpLabelList(corpus.LL_SOUND_CLASS, 1)],
        'avion2': [rt.ExpLabelList(corpus.LL_SOUND_CLASS, 1)],
        'bus1': [rt.ExpLabelList(corpus.LL_SOUND_CLASS, 1)],
        'metro_rouen22': [rt.ExpLabelList(corpus.LL_SOUND_CLASS, 1)],
    }

    EXPECTED_LABELS = {
        'avion1':   [rt.ExpLabel(corpus.LL_SOUND_CLASS, 'avion', 0, float('inf'))],
        'avion2': [rt.ExpLabel(corpus.LL_SOUND_CLASS, 'avion', 0, float('inf'))],
        'bus1':   [rt.ExpLabel(corpus.LL_SOUND_CLASS, 'bus', 0, float('inf'))],
        'metro_rouen22':   [rt.ExpLabel(corpus.LL_SOUND_CLASS, 'metro_rouen', 0, float('inf'))],
    }

    EXPECTED_NUMBER_OF_SUBVIEWS = 0

    def load(self):
        return io.RouenReader().load(self.SAMPLE_PATH)
