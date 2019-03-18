import os

import pytest
import requests_mock

from audiomate import corpus
from audiomate.corpus import io
from audiomate.corpus.io import esc
from tests import resources

from . import reader_test as rt


@pytest.fixture
def downloader():
    return io.ESC50Downloader()


@pytest.fixture()
def zip_data():
    path = resources.get_resource_path(
        ['sample_files', 'zip_sample_with_subfolder.zip']
    )
    with open(path, 'rb') as f:
        return f.read()


class TestESC50Downloader:

    def test_download(self, zip_data, downloader, tmpdir):
        target_folder = tmpdir.strpath

        with requests_mock.Mocker() as mock:
            mock.get(esc.DOWNLOAD_URL, content=zip_data)

            downloader.download(target_folder)

        assert os.path.isfile(os.path.join(target_folder, 'a.txt'))
        assert os.path.isfile(os.path.join(target_folder, 'subsub', 'b.txt'))
        assert os.path.isfile(os.path.join(target_folder, 'subsub', 'c.txt'))


class TestESC50Reader(rt.CorpusReaderTest):

    SAMPLE_PATH = resources.sample_corpus_path('esc50')
    FILE_TRACK_BASE_PATH = os.path.join(SAMPLE_PATH, 'audio')

    EXPECTED_NUMBER_OF_TRACKS = 10
    EXPECTED_TRACKS = [
        rt.ExpFileTrack('1-119125-A-45', '1-119125-A-45.wav'),
        rt.ExpFileTrack('1-12654-B-15', '1-12654-B-15.wav'),
        rt.ExpFileTrack('1-155858-E-25',  '1-155858-E-25.wav'),
        rt.ExpFileTrack('1-155858-F-25',  '1-155858-F-25.wav'),
        rt.ExpFileTrack('1-15689-A-4',  '1-15689-A-4.wav'),
        rt.ExpFileTrack('1-17124-A-43',  '1-17124-A-43.wav'),
        rt.ExpFileTrack('1-18755-A-4',  '1-18755-A-4.wav'),
        rt.ExpFileTrack('1-18755-B-4',  '1-18755-B-4.wav'),
        rt.ExpFileTrack('1-18757-A-4',  '1-18757-A-4.wav'),
        rt.ExpFileTrack('1-18810-A-49',  '1-18810-A-49.wav')
    ]

    EXPECTED_NUMBER_OF_UTTERANCES = 10
    EXPECTED_UTTERANCES = [
        rt.ExpUtterance('1-119125-A-45', '1-119125-A-45', None, 0, float('inf')),
        rt.ExpUtterance('1-12654-B-15', '1-12654-B-15', None, 0, float('inf')),
        rt.ExpUtterance('1-155858-E-25', '1-155858-E-25', None, 0, float('inf')),
        rt.ExpUtterance('1-155858-F-25', '1-155858-F-25', None, 0, float('inf')),
        rt.ExpUtterance('1-15689-A-4', '1-15689-A-4', None, 0, float('inf')),
        rt.ExpUtterance('1-17124-A-43', '1-17124-A-43', None, 0, float('inf')),
        rt.ExpUtterance('1-18755-A-4', '1-18755-A-4', None, 0, float('inf')),
        rt.ExpUtterance('1-18755-B-4', '1-18755-B-4', None, 0, float('inf')),
        rt.ExpUtterance('1-18757-A-4', '1-18757-A-4', None, 0, float('inf')),
        rt.ExpUtterance('1-18810-A-49', '1-18810-A-49', None, 0, float('inf')),
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 0

    EXPECTED_LABEL_LISTS = {
        '1-119125-A-45': [
            rt.ExpLabelList(corpus.LL_SOUND_CLASS, 1),
        ],
        '1-15689-A-4': [
            rt.ExpLabelList(corpus.LL_SOUND_CLASS, 1),
        ],
        '1-18810-A-49': [
            rt.ExpLabelList(corpus.LL_SOUND_CLASS, 1),
        ],
    }

    EXPECTED_LABELS = {
        '1-119125-A-45': [
            rt.ExpLabel(corpus.LL_SOUND_CLASS, 'train', 0, float('inf')),
        ],
        '1-15689-A-4': [
            rt.ExpLabel(corpus.LL_SOUND_CLASS, 'frog', 0, float('inf')),
        ],
        '1-18810-A-49': [
            rt.ExpLabel(corpus.LL_SOUND_CLASS, 'hand_saw', 0, float('inf')),
        ],
    }

    EXPECTED_NUMBER_OF_SUBVIEWS = 6
    EXPECTED_SUBVIEWS = [
        rt.ExpSubview('fold-1', ['1-119125-A-45', '1-12654-B-15',
                                 '1-155858-F-25', '1-17124-A-43',
                                 '1-18755-A-4']),
        rt.ExpSubview('fold-2', ['1-155858-E-25', '1-15689-A-4']),
        rt.ExpSubview('fold-3', ['1-18755-B-4']),
        rt.ExpSubview('fold-4', ['1-18757-A-4']),
        rt.ExpSubview('fold-5', ['1-18810-A-49']),
        rt.ExpSubview('esc-10', ['1-155858-E-25', '1-155858-F-25',
                                 '1-17124-A-43']),
    ]

    def load(self):
        return io.ESC50Reader().load(self.SAMPLE_PATH)
