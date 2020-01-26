import os

from audiomate import corpus
from audiomate import issuers
from audiomate.corpus import io
from audiomate.corpus.io import librispeech

import pytest
import requests_mock

from tests import resources
from . import reader_test as rt


class TestLibriSpeechDownloader:

    def test_download_all(self, tmpdir):
        target_folder = tmpdir.strpath
        downloader = io.LibriSpeechDownloader()

        with requests_mock.Mocker() as mock:
            # Return any size (doesn't matter, only for prints)
            for name, url in librispeech.SUBSETS.items():
                data_path = resources.get_resource_path([
                    'sample_archives',
                    'librispeech',
                    '{}.tar.gz'.format(name)
                ])
                with open(data_path, 'rb') as f:
                    data = f.read()
                mock.head(url, headers={'Content-Length': '100'})
                mock.get(url, content=data)

            downloader.download(target_folder)

        for name in librispeech.SUBSETS.keys():
            assert os.path.isdir(os.path.join(target_folder, name))

        assert os.path.isfile(os.path.join(target_folder, 'BOOKS.TXT'))
        assert os.path.isfile(os.path.join(target_folder, 'CHAPTERS.TXT'))
        assert os.path.isfile(os.path.join(target_folder, 'LICENSE.TXT'))
        assert os.path.isfile(os.path.join(target_folder, 'README.TXT'))
        assert os.path.isfile(os.path.join(target_folder, 'SPEAKERS.TXT'))

    def test_download_two_subsets(self, tmpdir):
        target_folder = tmpdir.strpath
        downloader = io.LibriSpeechDownloader(subsets=['dev-clean', 'test-clean'])

        with requests_mock.Mocker() as mock:
            # Return any size (doesn't matter, only for prints)
            for name, url in librispeech.SUBSETS.items():
                data_path = resources.get_resource_path([
                    'sample_archives',
                    'librispeech',
                    '{}.tar.gz'.format(name)
                ])
                with open(data_path, 'rb') as f:
                    data = f.read()
                mock.head(url, headers={'Content-Length': '100'})
                mock.get(url, content=data)

            downloader.download(target_folder)

        for name in ['dev-clean', 'test-clean']:
            assert os.path.isdir(os.path.join(target_folder, name))

        assert os.path.isfile(os.path.join(target_folder, 'BOOKS.TXT'))
        assert os.path.isfile(os.path.join(target_folder, 'CHAPTERS.TXT'))
        assert os.path.isfile(os.path.join(target_folder, 'LICENSE.TXT'))
        assert os.path.isfile(os.path.join(target_folder, 'README.TXT'))
        assert os.path.isfile(os.path.join(target_folder, 'SPEAKERS.TXT'))
