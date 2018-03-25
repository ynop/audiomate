import os

import pytest
import requests_mock

from pingu.corpus.io import voxforge

from tests import resources


@pytest.fixture()
def sample_tgz_content():
    with open(resources.sample_voxforge_file_path(), 'rb') as f:
        return f.read()


class TestVoxforgeDownloader:

    def test_download(self, tmpdir, sample_tgz_content):
        with requests_mock.Mocker() as mock:
            mock.get(voxforge.DOWNLOAD_URL['de'],
                     text='hallo <a href="test.tgz">test.tgz</a> blub')
            mock.get(os.path.join(voxforge.DOWNLOAD_URL['de'], 'test.tgz'), content=sample_tgz_content)

            downloader = voxforge.VoxforgeDownloader(lang='de')
            downloader.download(tmpdir.strpath)

        base_folder = os.path.join(tmpdir.strpath, 'Aaron-20080318-kdl')
        etc_folder = os.path.join(base_folder, 'etc')
        wav_folder = os.path.join(base_folder, 'wav')

        assert os.path.isfile(os.path.join(etc_folder, 'README'))

        assert os.path.isfile(os.path.join(wav_folder, 'b0019.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0020.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0021.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0022.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0023.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0024.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0025.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0026.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0027.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0028.wav'))

    def test_download_custom_url(self, tmpdir, sample_tgz_content):
        with requests_mock.Mocker() as mock:
            mock.get('http://someurl.com/some/download/dir',
                     text='hallo <a href="test.tgz">test.tgz</a> blub')
            mock.get('http://someurl.com/some/download/dir/test.tgz', content=sample_tgz_content)

            downloader = voxforge.VoxforgeDownloader(lang='de', url='http://someurl.com/some/download/dir')
            downloader.download(tmpdir.strpath)

        base_folder = os.path.join(tmpdir.strpath, 'Aaron-20080318-kdl')
        etc_folder = os.path.join(base_folder, 'etc')
        wav_folder = os.path.join(base_folder, 'wav')

        assert os.path.isfile(os.path.join(etc_folder, 'README'))

        assert os.path.isfile(os.path.join(wav_folder, 'b0019.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0020.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0021.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0022.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0023.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0024.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0025.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0026.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0027.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0028.wav'))

    def test_available_files(self):
        with requests_mock.Mocker() as mock:
            url = 'http://someurl.com/some/download/dir'
            mock.get(url, text=resources.sample_voxforge_response())
            files = voxforge.VoxforgeDownloader.available_files(url)

            assert set(files) == {
                'http://someurl.com/some/download/dir/1337ad-20170321-amr.tgz',
                'http://someurl.com/some/download/dir/1337ad-20170321-bej.tgz',
                'http://someurl.com/some/download/dir/1337ad-20170321-blf.tgz',
                'http://someurl.com/some/download/dir/1337ad-20170321-czb.tgz',
                'http://someurl.com/some/download/dir/1337ad-20170321-hii.tgz'
            }

    def test_download_files(self, tmpdir):
        files = [
            'http://someurl.com/some/download/dir/1337ad-20170321-amr.tgz',
            'http://someurl.com/some/download/dir/1337ad-20170321-bej.tgz',
            'http://someurl.com/some/download/dir/1337ad-20170321-blf.tgz',
            'http://someurl.com/some/download/dir/1337ad-20170321-czb.tgz',
            'http://someurl.com/some/download/dir/1337ad-20170321-hii.tgz'
        ]

        mock = requests_mock.Mocker()

        for file in files:
            mock.get(file, content='some content'.encode())

        file_paths = voxforge.VoxforgeDownloader.download_files(files, tmpdir.strpath)

        expected_file_paths = [
            os.path.join(tmpdir.strpath, '1337ad-20170321-amr.tgz'),
            os.path.join(tmpdir.strpath, '1337ad-20170321-bej.tgz'),
            os.path.join(tmpdir.strpath, '1337ad-20170321-blf.tgz'),
            os.path.join(tmpdir.strpath, '1337ad-20170321-czb.tgz'),
            os.path.join(tmpdir.strpath, '1337ad-20170321-hii.tgz')
        ]

        assert len(file_paths) == 5

        for file_path in expected_file_paths:
            assert os.path.isfile(file_path)
            assert file_path in file_paths

    def test_extract_files(self, tmpdir):
        extracted = voxforge.VoxforgeDownloader.extract_files([resources.sample_voxforge_file_path()], tmpdir.strpath)

        base_folder = os.path.join(tmpdir.strpath, 'Aaron-20080318-kdl')
        etc_folder = os.path.join(base_folder, 'etc')
        wav_folder = os.path.join(base_folder, 'wav')

        assert os.path.isfile(os.path.join(etc_folder, 'README'))

        assert os.path.isfile(os.path.join(wav_folder, 'b0019.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0020.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0021.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0022.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0023.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0024.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0025.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0026.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0027.wav'))
        assert os.path.isfile(os.path.join(wav_folder, 'b0028.wav'))

        assert len(extracted) == 1
        assert os.path.join(tmpdir.strpath, 'voxforge_sample') in extracted
