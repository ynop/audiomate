import os

from audiomate import corpus
from audiomate import issuers
from audiomate.corpus import io
from audiomate.corpus.io import voxforge

import pytest
import requests_mock

from tests import resources
from . import reader_test as rt


@pytest.fixture()
def sample_response():
    with open(resources.get_resource_path(['sample_files', 'voxforge_response.html']), 'r') as f:
        return f.read()


@pytest.fixture()
def sample_tgz_content():
    with open(resources.get_resource_path(['sample_files', 'voxforge_sample.tgz']), 'rb') as f:
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


class TestVoxforgeReader(rt.CorpusReaderTest):

    SAMPLE_PATH = resources.sample_corpus_path('voxforge')
    FILE_TRACK_BASE_PATH = SAMPLE_PATH

    EXPECTED_NUMBER_OF_TRACKS = 13
    EXPECTED_TRACKS = [
        rt.ExpFileTrack('1337ad-20170321-czb-de11-095', os.path.join('1337ad-20170321-czb',
                                                                     'wav',
                                                                     'de11-095.wav')),
        rt.ExpFileTrack('1337ad-20170321-czb-de11-096', os.path.join('1337ad-20170321-czb',
                                                                     'wav',
                                                                     'de11-096.wav')),
        rt.ExpFileTrack('anonymous-20081027-njq-a0479', os.path.join('anonymous-20081027-njq',
                                                                     'wav',
                                                                     'a0479.wav')),
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 4
    EXPECTED_ISSUERS = [
        rt.ExpSpeaker('1337ad', 5, issuers.Gender.FEMALE, issuers.AgeGroup.ADULT, 'deu'),
        rt.ExpSpeaker('anonymous-20081027-njq', 1, issuers.Gender.MALE, issuers.AgeGroup.ADULT, 'eng'),
        rt.ExpSpeaker('Katzer', 4, issuers.Gender.MALE, issuers.AgeGroup.YOUTH, 'eng'),
        rt.ExpSpeaker('knotyouraveragejo', 3, issuers.Gender.FEMALE, issuers.AgeGroup.ADULT, 'eng'),
    ]

    EXPECTED_NUMBER_OF_UTTERANCES = 13
    EXPECTED_UTTERANCES = [
        rt.ExpUtterance('1337ad-20170321-czb-de11-095', '1337ad-20170321-czb-de11-095',
                        '1337ad', 0, float('inf')),
        rt.ExpUtterance('1337ad-20170321-czb-de11-096', '1337ad-20170321-czb-de11-096',
                        '1337ad', 0, float('inf')),
        rt.ExpUtterance('anonymous-20081027-njq-a0479', 'anonymous-20081027-njq-a0479',
                        'anonymous-20081027-njq', 0, float('inf')),
    ]

    EXPECTED_LABEL_LISTS = {
        '1337ad-20170321-czb-de11-096': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT_RAW, 1),
        ],
        'Katzer-20140410-lyk-b0167': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT_RAW, 1),
        ],
        'anonymous-20081027-njq-a0479': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT_RAW, 1),
        ],
    }

    EXPECTED_LABELS = {
        '1337ad-20170321-czb-de11-096': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT,
                        'ES HANDELT SICH UM GETRENNTE RECHTSSYSTEME UND NUR EINES IST ANWENDBAR',
                        0, float('inf')),
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT_RAW,
                        'Es handelt sich um getrennte Rechtssysteme und nur eines ist anwendbar.',
                        0, float('inf')),
        ],
        'Katzer-20140410-lyk-b0167': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT,
                        'A LITTLE BEFORE DAWN OF THE DAY FOLLOWING THE FIRE RELIEF CAME',
                        0, float('inf')),
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT_RAW,
                        'A little before dawn of the day following, the fire relief came.',
                        0, float('inf')),
        ],
        'anonymous-20081027-njq-a0479': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT,
                        'I TRIED TO READ GEORGE MOORE LAST NIGHT AND WAS DREADFULLY BORED',
                        0, float('inf')),
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT_RAW,
                        'I tried to read George Moore last night, and was dreadfully bored.',
                        0, float('inf')),
        ],
    }

    EXPECTED_NUMBER_OF_SUBVIEWS = 0

    def load(self):
        return io.VoxforgeReader().load(self.SAMPLE_PATH)
