import os

import pytest
import requests_mock

from audiomate import corpus
from audiomate import issuers
from audiomate.corpus.io import common_voice

from tests import resources


@pytest.fixture()
def reader():
    return common_voice.CommonVoiceReader()


@pytest.fixture()
def tar_data():
    with open(resources.get_resource_path(['sample_files', 'cv_corpus_v1.tar.gz']), 'rb') as f:
        return f.read()


@pytest.fixture
def data_path():
    return resources.sample_corpus_path('common_voice')


class TestCommonVoiceDownloader:

    def test_download(self, tar_data, tmpdir):
        target_folder = tmpdir.strpath
        downloader = common_voice.CommonVoiceDownloader()

        with requests_mock.Mocker() as mock:
            mock.get(common_voice.DOWNLOAD_V1, content=tar_data)

            downloader.download(target_folder)

        assert os.path.isfile(os.path.join(target_folder, 'cv-valid-dev.csv'))
        assert os.path.isdir(os.path.join(target_folder, 'cv-valid-dev'))
        assert os.path.isfile(os.path.join(target_folder, 'cv-valid-train.csv'))
        assert os.path.isdir(os.path.join(target_folder, 'cv-valid-train'))


class TestCommonVoiceReader:

    def test_load_correct_number_of_tracks(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_tracks == 7

    @pytest.mark.parametrize('idx,path', [
        ('cv-valid-dev-sample-000000', os.path.join('cv-valid-dev', 'sample-000000.mp3')),
        ('cv-valid-dev-sample-000335', os.path.join('cv-valid-dev', 'sample-000335.mp3')),
        ('cv-valid-dev-sample-001879', os.path.join('cv-valid-dev', 'sample-001879.mp3')),
        ('cv-valid-dev-sample-004075', os.path.join('cv-valid-dev', 'sample-004075.mp3')),
        ('cv-valid-train-sample-000000', os.path.join('cv-valid-train', 'sample-000000.mp3')),
        ('cv-valid-train-sample-195733', os.path.join('cv-valid-train', 'sample-195733.mp3')),
        ('cv-valid-train-sample-195754', os.path.join('cv-valid-train', 'sample-195754.mp3'))
    ])
    def test_load_tracks(self, idx, path, reader, data_path):
        ds = reader.load(data_path)

        assert ds.tracks[idx].idx == idx
        assert ds.tracks[idx].path == os.path.join(data_path, path)

    def test_load_correct_number_of_speakers(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_issuers == 7

    @pytest.mark.parametrize('idx,age,gender,num_utt', [
        ('cv-valid-dev-sample-000000', issuers.AgeGroup.UNKNOWN, issuers.Gender.UNKNOWN, 1),
        ('cv-valid-dev-sample-000335', issuers.AgeGroup.ADULT, issuers.Gender.FEMALE, 1),
        ('cv-valid-dev-sample-001879', issuers.AgeGroup.UNKNOWN, issuers.Gender.UNKNOWN, 1),
        ('cv-valid-dev-sample-004075', issuers.AgeGroup.UNKNOWN, issuers.Gender.UNKNOWN, 1),
        ('cv-valid-train-sample-000000', issuers.AgeGroup.UNKNOWN, issuers.Gender.UNKNOWN, 1),
        ('cv-valid-train-sample-195733', issuers.AgeGroup.ADULT, issuers.Gender.MALE, 1),
        ('cv-valid-train-sample-195754', issuers.AgeGroup.UNKNOWN, issuers.Gender.UNKNOWN, 1)
    ])
    def test_load_issuers(self, idx, age, gender, num_utt, reader, data_path):
        ds = reader.load(data_path)

        assert ds.issuers[idx].idx == idx
        assert ds.issuers[idx].gender == gender
        assert ds.issuers[idx].age_group == age
        assert type(ds.issuers[idx]) == issuers.Speaker
        assert len(ds.issuers[idx].utterances) == num_utt

    def test_load_correct_number_of_utterances(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_utterances == 7

    @pytest.mark.parametrize('idx, issuer_idx', [
        ('cv-valid-dev-sample-000000', 'cv-valid-dev-sample-000000'),
        ('cv-valid-dev-sample-000335', 'cv-valid-dev-sample-000335'),
        ('cv-valid-dev-sample-001879', 'cv-valid-dev-sample-001879'),
        ('cv-valid-dev-sample-004075', 'cv-valid-dev-sample-004075'),
        ('cv-valid-train-sample-000000', 'cv-valid-train-sample-000000'),
        ('cv-valid-train-sample-195733', 'cv-valid-train-sample-195733'),
        ('cv-valid-train-sample-195754', 'cv-valid-train-sample-195754')
    ])
    def test_load_utterances(self, idx, issuer_idx, reader, data_path):
        ds = reader.load(data_path)

        assert ds.utterances[idx].idx == idx
        assert ds.utterances[idx].track.idx == idx
        assert ds.utterances[idx].issuer.idx == issuer_idx
        assert ds.utterances[idx].start == 0
        assert ds.utterances[idx].end == -1

    @pytest.mark.parametrize('idx, transcription', [
        ('cv-valid-dev-sample-000000', 'be careful with your prognostications said the stranger'),
        ('cv-valid-dev-sample-000335', 'love required them to stay with the people they loved'),
        ('cv-valid-dev-sample-001879', "who's down there with you"),
        ('cv-valid-dev-sample-004075', "the city sealer's office"),
        ('cv-valid-train-sample-000000', "trust in your heart but never forget that you're in the desert"),
        ('cv-valid-train-sample-195733', 'the battles may last for a long time perhaps even years'),
        ('cv-valid-train-sample-195754', 'the silicon sealant has dried')
    ])
    def test_load_transcription(self, idx, transcription, reader, data_path):
        ds = reader.load(data_path)

        ll = ds.utterances[idx].label_lists[corpus.LL_WORD_TRANSCRIPT]

        assert len(ll) == 1
        assert ll[0].value == transcription
        assert ll[0].start == 0
        assert ll[0].end == -1

    def test_load_correct_number_of_subviews(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_subviews == 2

    @pytest.mark.parametrize('idx, utts', [
        ('cv-valid-dev', ['cv-valid-dev-sample-000000', 'cv-valid-dev-sample-000335',
                          'cv-valid-dev-sample-001879', 'cv-valid-dev-sample-004075']),
        ('cv-valid-train', ['cv-valid-train-sample-000000', 'cv-valid-train-sample-195733',
                            'cv-valid-train-sample-195754'])
    ])
    def test_subviews(self, idx, utts, reader, data_path):
        ds = reader.load(data_path)

        assert idx in ds.subviews.keys()
        assert ds.subviews[idx].num_utterances == len(utts)
        assert set(ds.subviews[idx].utterances.keys()) == set(utts)
