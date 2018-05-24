import os

import pytest
import requests_mock

from audiomate.corpus.io import free_spoken_digits
from audiomate.corpus import assets

from tests import resources


@pytest.fixture()
def reader():
    return free_spoken_digits.FreeSpokenDigitReader()


@pytest.fixture()
def zip_data():
    with open(resources.get_resource_path(['sample_files', 'zip_sample_with_subfolder.zip']), 'rb') as f:
        return f.read()


@pytest.fixture
def data_path():
    return resources.sample_corpus_path('free_spoken_digits')


class TestFreeSpokenDigitDownloader:

    def test_download(self, zip_data, tmpdir):
        target_folder = tmpdir.strpath
        downloader = free_spoken_digits.FreeSpokenDigitDownloader()

        with requests_mock.Mocker() as mock:
            mock.get(free_spoken_digits.MASTER_DOWNLOAD_URL, content=zip_data)

            downloader.download(target_folder)

        assert os.path.isfile(os.path.join(target_folder, 'a.txt'))
        assert os.path.isfile(os.path.join(target_folder, 'subsub', 'b.txt'))
        assert os.path.isfile(os.path.join(target_folder, 'subsub', 'c.txt'))


class TestFreeSpokenDigitReader:

    def test_load_correct_number_of_files(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_files == 4

    @pytest.mark.parametrize('idx,path', [
        ('0_jackson_0', os.path.join('recordings', '0_jackson_0.wav')),
        ('1_jackson_0', os.path.join('recordings', '1_jackson_0.wav')),
        ('2_theo_0', os.path.join('recordings', '2_theo_0.wav')),
        ('2_theo_1', os.path.join('recordings', '2_theo_1.wav')),
    ])
    def test_load_files(self, idx, path, reader, data_path):
        ds = reader.load(data_path)

        assert ds.files[idx].idx == idx
        assert ds.files[idx].path == os.path.join(data_path, path)

    def test_load_correct_number_of_speakers(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_issuers == 2

    @pytest.mark.parametrize('idx,num_utt', [
        ('jackson', 2),
        ('theo', 2)
    ])
    def test_load_issuers(self, idx, num_utt, reader, data_path):
        ds = reader.load(data_path)

        assert ds.issuers[idx].idx == idx
        assert type(ds.issuers[idx]) == assets.Speaker
        assert len(ds.issuers[idx].utterances) == num_utt

    def test_load_correct_number_of_utterances(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_utterances == 4

    @pytest.mark.parametrize('idx, issuer_idx', [
        ('0_jackson_0', 'jackson'),
        ('1_jackson_0', 'jackson'),
        ('2_theo_0', 'theo'),
        ('2_theo_1', 'theo')
    ])
    def test_load_utterances(self, idx, issuer_idx, reader, data_path):
        ds = reader.load(data_path)

        assert ds.utterances[idx].idx == idx
        assert ds.utterances[idx].file.idx == idx
        assert ds.utterances[idx].issuer.idx == issuer_idx
        assert ds.utterances[idx].start == 0
        assert ds.utterances[idx].end == -1

    @pytest.mark.parametrize('idx, transcription', [
        ('0_jackson_0', '0'),
        ('1_jackson_0', '1'),
        ('2_theo_0', '2'),
        ('2_theo_1', '2')
    ])
    def test_load_transcription(self, idx, transcription, reader, data_path):
        ds = reader.load(data_path)

        ll = ds.utterances[idx].label_lists['transcription']

        assert len(ll) == 1
        assert ll[0].value == transcription
        assert ll[0].start == 0
        assert ll[0].end == -1
