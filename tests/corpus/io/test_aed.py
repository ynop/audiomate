import os

import requests_mock
import pytest

from audiomate import corpus
from audiomate.corpus import io
from audiomate.corpus.io import aed

from tests import resources
from . import reader_test as rt


@pytest.fixture
def downloader():
    return io.AEDDownloader()


@pytest.fixture()
def zip_data():
    with open(resources.get_resource_path(['sample_files', 'zip_sample_with_subfolder.zip']), 'rb') as f:
        return f.read()


class TestAEDDownloader:

    def test_download(self, zip_data, downloader, tmpdir):
        target_folder = tmpdir.strpath

        with requests_mock.Mocker() as mock:
            mock.get(aed.DATA_URL, content=zip_data)

            downloader.download(target_folder)

        assert os.path.isfile(os.path.join(target_folder, 'a.txt'))
        assert os.path.isfile(os.path.join(target_folder, 'subsub', 'b.txt'))
        assert os.path.isfile(os.path.join(target_folder, 'subsub', 'c.txt'))


class TestAEDReader(rt.CorpusReaderTest):

    SAMPLE_PATH = resources.sample_corpus_path('aed')
    FILE_TRACK_BASE_PATH = SAMPLE_PATH

    EXPECTED_NUMBER_OF_TRACKS = 10
    EXPECTED_TRACKS = [
        rt.ExpFileTrack('acoustic_guitar_16', os.path.join('test', 'acoustic_guitar_16.wav')),
        rt.ExpFileTrack('footstep_300', os.path.join('test', 'footstep_300.wav')),
        rt.ExpFileTrack('violin_36', os.path.join('test', 'violin_36.wav')),
        rt.ExpFileTrack('airplane_1', os.path.join('train', 'airplane', 'airplane_1.wav')),
        rt.ExpFileTrack('airplane_23', os.path.join('train', 'airplane', 'airplane_23.wav')),
        rt.ExpFileTrack('airplane_33', os.path.join('train', 'airplane', 'airplane_33.wav')),
        rt.ExpFileTrack('footstep_16', os.path.join('train', 'footstep', 'footstep_16.wav')),
        rt.ExpFileTrack('helicopter_9', os.path.join('train', 'helicopter', 'helicopter_9.wav')),
        rt.ExpFileTrack('tone_12', os.path.join('train', 'tone', 'tone_12.wav')),
        rt.ExpFileTrack('tone_35', os.path.join('train', 'tone', 'tone_35.wav')),
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 0

    EXPECTED_NUMBER_OF_UTTERANCES = 10
    EXPECTED_UTTERANCES = [
        rt.ExpUtterance('acoustic_guitar_16', 'acoustic_guitar_16', None, 0, float('inf')),
        rt.ExpUtterance('footstep_300', 'footstep_300', None, 0, float('inf')),
        rt.ExpUtterance('violin_36', 'violin_36', None, 0, float('inf')),
        rt.ExpUtterance('airplane_1', 'airplane_1', None, 0, float('inf')),
        rt.ExpUtterance('airplane_23', 'airplane_23', None, 0, float('inf')),
        rt.ExpUtterance('airplane_33', 'airplane_33', None, 0, float('inf')),
        rt.ExpUtterance('footstep_16', 'footstep_16', None, 0, float('inf')),
        rt.ExpUtterance('helicopter_9', 'helicopter_9', None, 0, float('inf')),
        rt.ExpUtterance('tone_12', 'tone_12', None, 0, float('inf')),
        rt.ExpUtterance('tone_35', 'tone_35', None, 0, float('inf')),
    ]

    EXPECTED_LABEL_LISTS = {
        'airplane_23': [rt.ExpLabelList(corpus.LL_SOUND_CLASS, 1)],
        'tone_12': [rt.ExpLabelList(corpus.LL_SOUND_CLASS, 1)],
        'acoustic_guitar_16': [rt.ExpLabelList(corpus.LL_SOUND_CLASS, 1)],
    }

    EXPECTED_LABELS = {
        'airplane_23': [
            rt.ExpLabel(corpus.LL_SOUND_CLASS, 'airplane', 0, float('inf')),
        ],
        'tone_12': [
            rt.ExpLabel(corpus.LL_SOUND_CLASS, 'tone', 0, float('inf')),
        ],
        'acoustic_guitar_16': [
            rt.ExpLabel(corpus.LL_SOUND_CLASS, 'acoustic_guitar', 0, float('inf')),
        ],
    }

    EXPECTED_NUMBER_OF_SUBVIEWS = 2
    EXPECTED_SUBVIEWS = [
        rt.ExpSubview('train', [
            'airplane_1',
            'airplane_23',
            'airplane_33',
            'footstep_16',
            'helicopter_9',
            'tone_12',
            'tone_35',
        ]),
        rt.ExpSubview('test', [
            'acoustic_guitar_16',
            'footstep_300',
            'violin_36',
        ]),
    ]

    def load(self):
        reader = io.AEDReader()
        return reader.load(self.SAMPLE_PATH)
