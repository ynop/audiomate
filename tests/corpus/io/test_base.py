import os

from audiomate.corpus.io import base

import pytest


def create_mock_corpus_downloader():

    class MockCorpusDownloader(base.CorpusDownloader):

        @classmethod
        def type(cls):
            return 'mock'

        def _download(self, target_path):
            os.makedirs(os.path.join(target_path, 'subfolder', 'a.txt'))

    return MockCorpusDownloader()


class TestCorpusDownloader:

    def test_force_redownload_overwrites_existing_directory(self, tmpdir):
        target_folder = tmpdir.strpath
        corpus_dl = create_mock_corpus_downloader()

        tmpdir.mkdir('subfolder').join('b.txt')
        corpus_dl.download(target_folder, force_redownload=True)

        assert len(os.listdir(tmpdir)) == 1
        assert os.path.exists(os.path.join(target_folder, 'subfolder', 'a.txt'))
        assert not os.path.exists(os.path.join(target_folder, 'subfolder', 'b.txt'))

    def test_existing_directory_forces_io_error(self, tmpdir):
        target_folder = tmpdir.strpath
        corpus_dl = create_mock_corpus_downloader()

        tmpdir.mkdir('subfolder').join('a.txt')

        with pytest.raises(IOError):
            corpus_dl.download(target_folder, force_redownload=False)


def create_mock_corpus_reader():

    class MockCorpusReader(base.CorpusReader):

        @classmethod
        def type(cls):
            return 'mock'

        def _load(self, path):
            return

        def _check_for_missing_files(self, path):
            return []

    return MockCorpusReader()


class TestCorpusReader:

    def test_invalid_path_forces_io_error(self, tmpdir):
        target_folder = tmpdir.strpath
        corpus_reader = create_mock_corpus_reader()

        with pytest.raises(IOError):
            path = os.path.join(target_folder, 'tmp')
            corpus_reader.load(path)
