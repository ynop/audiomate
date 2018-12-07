import os

from audiomate.corpus.io import downloader

import pytest
import requests_mock

from tests import resources


MOCK_URL = 'https://downloadings.dl/download'


@pytest.fixture()
def tar_data():
    path = resources.get_resource_path([
        'sample_files',
        'cv_corpus_v1.tar.gz'
    ])

    with open(path, 'rb') as f:
        return f.read()


@pytest.fixture()
def zip_data():
    path = resources.get_resource_path([
        'sample_files',
        'zip_sample_with_subfolder.zip'
    ])

    with open(path, 'rb') as f:
        return f.read()


class MockArchiveDownloader(downloader.ArchiveDownloader):

    def type(self):
        return 'mock'


class TestArchiveDownloader:

    def test_download_tar(self, tar_data, tmpdir):
        target_folder = tmpdir.strpath
        corpus_dl = MockArchiveDownloader(
            MOCK_URL,
            downloader.ArkType.TAR
        )

        with requests_mock.Mocker() as mock:
            mock.get(MOCK_URL, content=tar_data)
            corpus_dl.download(target_folder)

        base_folder = os.path.join(target_folder, 'common_voice')
        assert os.path.isdir(base_folder)

        assert os.path.isfile(os.path.join(base_folder, 'cv-valid-dev.csv'))
        assert os.path.isdir(os.path.join(base_folder, 'cv-valid-dev'))
        assert os.path.isfile(os.path.join(base_folder, 'cv-valid-train.csv'))
        assert os.path.isdir(os.path.join(base_folder, 'cv-valid-train'))

    def test_download_zip(self, zip_data, tmpdir):
        target_folder = tmpdir.strpath
        corpus_dl = MockArchiveDownloader(
            MOCK_URL,
            downloader.ArkType.ZIP
        )

        with requests_mock.Mocker() as mock:
            mock.get(MOCK_URL, content=zip_data)
            corpus_dl.download(target_folder)

        base_folder = os.path.join(target_folder, 'subfolder')
        assert os.path.isdir(base_folder)

        assert os.path.isfile(os.path.join(base_folder, 'a.txt'))
        assert os.path.isfile(os.path.join(base_folder, 'subsub', 'b.txt'))
        assert os.path.isfile(os.path.join(base_folder, 'subsub', 'c.txt'))

    def test_download_auto(self, zip_data, tmpdir):
        target_folder = tmpdir.strpath
        corpus_dl = MockArchiveDownloader(
            MOCK_URL,
            downloader.ArkType.AUTO
        )

        with requests_mock.Mocker() as mock:
            mock.get(MOCK_URL, content=zip_data)
            corpus_dl.download(target_folder)

        base_folder = os.path.join(target_folder, 'subfolder')
        assert os.path.isdir(base_folder)

        assert os.path.isfile(os.path.join(base_folder, 'a.txt'))
        assert os.path.isfile(os.path.join(base_folder, 'subsub', 'b.txt'))
        assert os.path.isfile(os.path.join(base_folder, 'subsub', 'c.txt'))

    def test_download_moves_files_up(self, zip_data, tmpdir):
        target_folder = tmpdir.strpath
        corpus_dl = MockArchiveDownloader(
            MOCK_URL,
            downloader.ArkType.AUTO,
            move_files_up=True
        )

        with requests_mock.Mocker() as mock:
            mock.get(MOCK_URL, content=zip_data)
            corpus_dl.download(target_folder)

        assert os.path.isfile(os.path.join(target_folder, 'a.txt'))
        assert os.path.isfile(os.path.join(target_folder, 'subsub', 'b.txt'))
        assert os.path.isfile(os.path.join(target_folder, 'subsub', 'c.txt'))
