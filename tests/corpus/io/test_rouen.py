import os

import pytest
import requests_mock

from audiomate import corpus
from audiomate.corpus import io
from audiomate.corpus.io import rouen
from tests import resources


@pytest.fixture
def reader():
    return io.RouenReader()


@pytest.fixture
def downloader():
    return io.RouenDownloader()


@pytest.fixture
def data_path():
    return resources.sample_corpus_path('rouen')


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


class TestRouenReader:

    def test_load_tracks(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_tracks == 4

        assert ds.tracks['avion1'].idx == 'avion1'
        assert ds.tracks['avion1'].path == os.path.join(data_path, 'avion1.wav')

        assert ds.tracks['avion2'].idx == 'avion2'
        assert ds.tracks['avion2'].path == os.path.join(data_path, 'avion2.wav')

        assert ds.tracks['bus1'].idx == 'bus1'
        assert ds.tracks['bus1'].path == os.path.join(data_path, 'bus1.wav')

        assert ds.tracks['metro_rouen22'].idx == 'metro_rouen22'
        assert ds.tracks['metro_rouen22'].path == os.path.join(data_path, 'metro_rouen22.wav')

    def test_load_utterances(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_utterances == 4

        assert ds.utterances['avion1'].idx == 'avion1'
        assert ds.utterances['avion1'].track.idx == 'avion1'
        assert ds.utterances['avion1'].issuer is None
        assert ds.utterances['avion1'].start == 0
        assert ds.utterances['avion1'].end == -1

        assert ds.utterances['avion2'].idx == 'avion2'
        assert ds.utterances['avion2'].track.idx == 'avion2'
        assert ds.utterances['avion2'].issuer is None
        assert ds.utterances['avion2'].start == 0
        assert ds.utterances['avion2'].end == -1

        assert ds.utterances['bus1'].idx == 'bus1'
        assert ds.utterances['bus1'].track.idx == 'bus1'
        assert ds.utterances['bus1'].issuer is None
        assert ds.utterances['bus1'].start == 0
        assert ds.utterances['bus1'].end == -1

        assert ds.utterances['metro_rouen22'].idx == 'metro_rouen22'
        assert ds.utterances['metro_rouen22'].track.idx == 'metro_rouen22'
        assert ds.utterances['metro_rouen22'].issuer is None
        assert ds.utterances['metro_rouen22'].start == 0
        assert ds.utterances['metro_rouen22'].end == -1

    def test_load_issuers(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_issuers == 0

    def test_load_labels(self, reader, data_path):
        ds = reader.load(data_path)

        ll = ds.utterances['avion1'].label_lists[corpus.LL_SOUND_CLASS]
        assert len(ll) == 1
        assert ll.labels[0].value == 'avion'

        ll = ds.utterances['avion2'].label_lists[corpus.LL_SOUND_CLASS]
        assert len(ll) == 1
        assert ll.labels[0].value == 'avion'

        ll = ds.utterances['bus1'].label_lists[corpus.LL_SOUND_CLASS]
        assert len(ll) == 1
        assert ll.labels[0].value == 'bus'

        ll = ds.utterances['metro_rouen22'].label_lists[corpus.LL_SOUND_CLASS]
        assert len(ll) == 1
        assert ll.labels[0].value == 'metro_rouen'
