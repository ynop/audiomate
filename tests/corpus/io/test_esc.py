import os

import pytest
import requests_mock

from audiomate.corpus import io
from audiomate.corpus.io import esc
from tests import resources


@pytest.fixture
def reader():
    return io.ESC50Reader()


@pytest.fixture
def downloader():
    return io.ESC50Downloader()


@pytest.fixture
def data_path():
    return resources.sample_corpus_path('esc50')


@pytest.fixture()
def zip_data():
    with open(resources.get_resource_path(['sample_files', 'zip_sample_with_subfolder.zip']), 'rb') as f:
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


class TestESC50Reader:

    def test_load_files(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_files == 10

        assert ds.files['1-119125-A-45'].idx == '1-119125-A-45'
        assert ds.files['1-119125-A-45'].path == os.path.join(data_path, 'audio', '1-119125-A-45.wav')

        assert ds.files['1-12654-B-15'].idx == '1-12654-B-15'
        assert ds.files['1-12654-B-15'].path == os.path.join(data_path, 'audio', '1-12654-B-15.wav')

        assert ds.files['1-155858-E-25'].idx == '1-155858-E-25'
        assert ds.files['1-155858-E-25'].path == os.path.join(data_path, 'audio', '1-155858-E-25.wav')

        assert ds.files['1-155858-F-25'].idx == '1-155858-F-25'
        assert ds.files['1-155858-F-25'].path == os.path.join(data_path, 'audio', '1-155858-F-25.wav')

        assert ds.files['1-15689-A-4'].idx == '1-15689-A-4'
        assert ds.files['1-15689-A-4'].path == os.path.join(data_path, 'audio', '1-15689-A-4.wav')

        assert ds.files['1-17124-A-43'].idx == '1-17124-A-43'
        assert ds.files['1-17124-A-43'].path == os.path.join(data_path, 'audio', '1-17124-A-43.wav')

        assert ds.files['1-18755-A-4'].idx == '1-18755-A-4'
        assert ds.files['1-18755-A-4'].path == os.path.join(data_path, 'audio', '1-18755-A-4.wav')

        assert ds.files['1-18755-B-4'].idx == '1-18755-B-4'
        assert ds.files['1-18755-B-4'].path == os.path.join(data_path, 'audio', '1-18755-B-4.wav')

        assert ds.files['1-18757-A-4'].idx == '1-18757-A-4'
        assert ds.files['1-18757-A-4'].path == os.path.join(data_path, 'audio', '1-18757-A-4.wav')

        assert ds.files['1-18810-A-49'].idx == '1-18810-A-49'
        assert ds.files['1-18810-A-49'].path == os.path.join(data_path, 'audio', '1-18810-A-49.wav')

    def test_load_utterances(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_utterances == 10

        assert ds.utterances['1-119125-A-45'].idx == '1-119125-A-45'
        assert ds.utterances['1-119125-A-45'].file.idx == '1-119125-A-45'
        assert ds.utterances['1-119125-A-45'].issuer is None
        assert ds.utterances['1-119125-A-45'].start == 0
        assert ds.utterances['1-119125-A-45'].end == -1

        assert ds.utterances['1-12654-B-15'].idx == '1-12654-B-15'
        assert ds.utterances['1-12654-B-15'].file.idx == '1-12654-B-15'
        assert ds.utterances['1-12654-B-15'].issuer is None
        assert ds.utterances['1-12654-B-15'].start == 0
        assert ds.utterances['1-12654-B-15'].end == -1

        assert ds.utterances['1-155858-E-25'].idx == '1-155858-E-25'
        assert ds.utterances['1-155858-E-25'].file.idx == '1-155858-E-25'
        assert ds.utterances['1-155858-E-25'].issuer is None
        assert ds.utterances['1-155858-E-25'].start == 0
        assert ds.utterances['1-155858-E-25'].end == -1

        assert ds.utterances['1-155858-F-25'].idx == '1-155858-F-25'
        assert ds.utterances['1-155858-F-25'].file.idx == '1-155858-F-25'
        assert ds.utterances['1-155858-F-25'].issuer is None
        assert ds.utterances['1-155858-F-25'].start == 0
        assert ds.utterances['1-155858-F-25'].end == -1

        assert ds.utterances['1-15689-A-4'].idx == '1-15689-A-4'
        assert ds.utterances['1-15689-A-4'].file.idx == '1-15689-A-4'
        assert ds.utterances['1-15689-A-4'].issuer is None
        assert ds.utterances['1-15689-A-4'].start == 0
        assert ds.utterances['1-15689-A-4'].end == -1

        assert ds.utterances['1-17124-A-43'].idx == '1-17124-A-43'
        assert ds.utterances['1-17124-A-43'].file.idx == '1-17124-A-43'
        assert ds.utterances['1-17124-A-43'].issuer is None
        assert ds.utterances['1-17124-A-43'].start == 0
        assert ds.utterances['1-17124-A-43'].end == -1

        assert ds.utterances['1-18755-A-4'].idx == '1-18755-A-4'
        assert ds.utterances['1-18755-A-4'].file.idx == '1-18755-A-4'
        assert ds.utterances['1-18755-A-4'].issuer is None
        assert ds.utterances['1-18755-A-4'].start == 0
        assert ds.utterances['1-18755-A-4'].end == -1

        assert ds.utterances['1-18755-B-4'].idx == '1-18755-B-4'
        assert ds.utterances['1-18755-B-4'].file.idx == '1-18755-B-4'
        assert ds.utterances['1-18755-B-4'].issuer is None
        assert ds.utterances['1-18755-B-4'].start == 0
        assert ds.utterances['1-18755-B-4'].end == -1

        assert ds.utterances['1-18757-A-4'].idx == '1-18757-A-4'
        assert ds.utterances['1-18757-A-4'].file.idx == '1-18757-A-4'
        assert ds.utterances['1-18757-A-4'].issuer is None
        assert ds.utterances['1-18757-A-4'].start == 0
        assert ds.utterances['1-18757-A-4'].end == -1

        assert ds.utterances['1-18810-A-49'].idx == '1-18810-A-49'
        assert ds.utterances['1-18810-A-49'].file.idx == '1-18810-A-49'
        assert ds.utterances['1-18810-A-49'].issuer is None
        assert ds.utterances['1-18810-A-49'].start == 0
        assert ds.utterances['1-18810-A-49'].end == -1

    def test_load_issuers(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_issuers == 0

    def test_load_labels(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.utterances['1-119125-A-45'].label_lists['default'].labels[0].value == 'train'
        assert ds.utterances['1-119125-A-45'].label_lists['default'].labels[0].start == 0
        assert ds.utterances['1-119125-A-45'].label_lists['default'].labels[0].end == -1

        assert ds.utterances['1-15689-A-4'].label_lists['default'].labels[0].value == 'frog'
        assert ds.utterances['1-15689-A-4'].label_lists['default'].labels[0].start == 0
        assert ds.utterances['1-15689-A-4'].label_lists['default'].labels[0].end == -1

        assert ds.utterances['1-18810-A-49'].label_lists['default'].labels[0].value == 'hand_saw'
        assert ds.utterances['1-18810-A-49'].label_lists['default'].labels[0].start == 0
        assert ds.utterances['1-18810-A-49'].label_lists['default'].labels[0].end == -1

    def test_load_fold_subsets(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.subviews['fold-1'].num_utterances == 5
        assert set(ds.subviews['fold-1'].utterances.keys()) == {'1-119125-A-45', '1-12654-B-15', '1-155858-F-25',
                                                                '1-17124-A-43', '1-18755-A-4'}
        assert ds.subviews['fold-2'].num_utterances == 2
        assert set(ds.subviews['fold-2'].utterances.keys()) == {'1-155858-E-25', '1-15689-A-4'}
        assert ds.subviews['fold-3'].num_utterances == 1
        assert set(ds.subviews['fold-3'].utterances.keys()) == {'1-18755-B-4'}
        assert ds.subviews['fold-4'].num_utterances == 1
        assert set(ds.subviews['fold-4'].utterances.keys()) == {'1-18757-A-4'}
        assert ds.subviews['fold-5'].num_utterances == 1
        assert set(ds.subviews['fold-5'].utterances.keys()) == {'1-18810-A-49'}

    def test_load_esc_10_subset(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.subviews['esc-10'].num_utterances == 3
        assert set(ds.subviews['esc-10'].utterances.keys()) == {'1-155858-E-25', '1-155858-F-25', '1-17124-A-43'}
