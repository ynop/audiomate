import os

import pytest
import requests_mock

from audiomate.corpus.io import voxforge
from audiomate.corpus import assets

from tests import resources


@pytest.fixture()
def sample_response():
    with open(resources.get_resource_path(['sample_files', 'voxforge_response.html']), 'r') as f:
        return f.read()


@pytest.fixture()
def sample_tgz_content():
    with open(resources.get_resource_path(['sample_files', 'voxforge_sample.tgz']), 'rb') as f:
        return f.read()


@pytest.fixture()
def reader():
    return voxforge.VoxforgeReader()


@pytest.fixture()
def sample_corpus_path():
    return resources.sample_corpus_path('voxforge')


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

    def test_available_files(self, sample_response):
        with requests_mock.Mocker() as mock:
            url = 'http://someurl.com/some/download/dir'
            mock.get(url, text=sample_response)
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

        with requests_mock.Mocker() as mock:
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
        sample_file_path = resources.get_resource_path(['sample_files', 'voxforge_sample.tgz'])
        extracted = voxforge.VoxforgeDownloader.extract_files([sample_file_path], tmpdir.strpath)

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


class TestVoxforgeReader:

    def test_load_files(self, reader, sample_corpus_path):
        ds = reader.load(sample_corpus_path)

        assert ds.num_files == 13

        assert ds.files['1337ad-20170321-czb-de11-095'].idx == '1337ad-20170321-czb-de11-095'
        assert ds.files['1337ad-20170321-czb-de11-095'].path == os.path.join(sample_corpus_path,
                                                                             '1337ad-20170321-czb',
                                                                             'wav',
                                                                             'de11-095.wav')

        assert ds.files['1337ad-20170321-czb-de11-096'].idx == '1337ad-20170321-czb-de11-096'
        assert ds.files['1337ad-20170321-czb-de11-096'].path == os.path.join(sample_corpus_path,
                                                                             '1337ad-20170321-czb',
                                                                             'wav',
                                                                             'de11-096.wav')

        assert ds.files['anonymous-20081027-njq-a0479'].idx == 'anonymous-20081027-njq-a0479'
        assert ds.files['anonymous-20081027-njq-a0479'].path == os.path.join(sample_corpus_path,
                                                                             'anonymous-20081027-njq',
                                                                             'wav',
                                                                             'a0479.wav')

    def test_load_issuers(self, reader, sample_corpus_path):
        ds = reader.load(sample_corpus_path)

        assert ds.num_issuers == 4

        assert ds.issuers['1337ad'].idx == '1337ad'
        assert type(ds.issuers['1337ad']) == assets.Speaker
        assert ds.issuers['1337ad'].gender == assets.Gender.FEMALE
        assert ds.issuers['1337ad'].age_group == assets.AgeGroup.ADULT
        assert ds.issuers['1337ad'].native_language == 'deu'

        assert ds.issuers['anonymous-20081027-njq'].idx == 'anonymous-20081027-njq'
        assert type(ds.issuers['anonymous-20081027-njq']) == assets.Speaker
        assert ds.issuers['anonymous-20081027-njq'].gender == assets.Gender.MALE
        assert ds.issuers['anonymous-20081027-njq'].age_group == assets.AgeGroup.ADULT
        assert ds.issuers['anonymous-20081027-njq'].native_language == 'eng'

        assert ds.issuers['Katzer'].idx == 'Katzer'
        assert type(ds.issuers['Katzer']) == assets.Speaker
        assert ds.issuers['Katzer'].gender == assets.Gender.MALE
        assert ds.issuers['Katzer'].age_group == assets.AgeGroup.YOUTH
        assert ds.issuers['Katzer'].native_language == 'eng'

        assert ds.issuers['knotyouraveragejo'].idx == 'knotyouraveragejo'
        assert type(ds.issuers['knotyouraveragejo']) == assets.Speaker
        assert ds.issuers['knotyouraveragejo'].gender == assets.Gender.FEMALE
        assert ds.issuers['knotyouraveragejo'].age_group == assets.AgeGroup.ADULT
        assert ds.issuers['knotyouraveragejo'].native_language == 'eng'

    def test_load_utterances(self, reader, sample_corpus_path):
        ds = reader.load(sample_corpus_path)

        assert ds.num_utterances == 13

        assert ds.utterances['1337ad-20170321-czb-de11-095'].idx == '1337ad-20170321-czb-de11-095'
        assert ds.utterances['1337ad-20170321-czb-de11-095'].file.idx == '1337ad-20170321-czb-de11-095'
        assert ds.utterances['1337ad-20170321-czb-de11-095'].start == 0
        assert ds.utterances['1337ad-20170321-czb-de11-095'].end == -1
        assert ds.utterances['1337ad-20170321-czb-de11-095'].issuer.idx == '1337ad'

        assert ds.utterances['1337ad-20170321-czb-de11-096'].idx == '1337ad-20170321-czb-de11-096'
        assert ds.utterances['1337ad-20170321-czb-de11-096'].file.idx == '1337ad-20170321-czb-de11-096'
        assert ds.utterances['1337ad-20170321-czb-de11-096'].start == 0
        assert ds.utterances['1337ad-20170321-czb-de11-096'].end == -1
        assert ds.utterances['1337ad-20170321-czb-de11-096'].issuer.idx == '1337ad'

        assert ds.utterances['anonymous-20081027-njq-a0479'].idx == 'anonymous-20081027-njq-a0479'
        assert ds.utterances['anonymous-20081027-njq-a0479'].file.idx == 'anonymous-20081027-njq-a0479'
        assert ds.utterances['anonymous-20081027-njq-a0479'].start == 0
        assert ds.utterances['anonymous-20081027-njq-a0479'].end == -1
        assert ds.utterances['anonymous-20081027-njq-a0479'].issuer.idx == 'anonymous-20081027-njq'

    def test_load_transcriptions(self, reader, sample_corpus_path):
        ds = reader.load(sample_corpus_path)

        assert ds.utterances['1337ad-20170321-czb-de11-096'].label_lists['transcription'][0].value == \
            'ES HANDELT SICH UM GETRENNTE RECHTSSYSTEME UND NUR EINES IST ANWENDBAR'
        assert ds.utterances['1337ad-20170321-czb-de11-096'].label_lists['transcription_raw'][0].value == \
            'Es handelt sich um getrennte Rechtssysteme und nur eines ist anwendbar.'

        assert ds.utterances['Katzer-20140410-lyk-b0167'].label_lists['transcription'][0].value == \
            'A LITTLE BEFORE DAWN OF THE DAY FOLLOWING THE FIRE RELIEF CAME'
        assert ds.utterances['Katzer-20140410-lyk-b0167'].label_lists['transcription_raw'][0].value == \
            'A little before dawn of the day following, the fire relief came.'

        assert ds.utterances['anonymous-20081027-njq-a0479'].label_lists['transcription'][0].value == \
            'I TRIED TO READ GEORGE MOORE LAST NIGHT AND WAS DREADFULLY BORED'
        assert ds.utterances['anonymous-20081027-njq-a0479'].label_lists['transcription_raw'][0].value == \
            'I tried to read George Moore last night, and was dreadfully bored.'
