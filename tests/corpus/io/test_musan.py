import os

from audiomate import corpus
from audiomate import issuers
from audiomate.corpus import io
from audiomate.corpus.io import musan

import pytest
import requests_mock

from tests import resources


@pytest.fixture()
def tar_data():
    with open(resources.get_resource_path(['sample_files', 'cv_corpus_v1.tar.gz']), 'rb') as f:
        return f.read()


@pytest.fixture()
def reader():
    return io.MusanReader()


@pytest.fixture()
def sample_path():
    return resources.sample_corpus_path('musan')


class TestMusanDownloader:

    def test_download(self, tar_data, tmpdir):
        target_folder = tmpdir.strpath
        downloader = io.MusanDownloader()

        with requests_mock.Mocker() as mock:
            mock.get(musan.DOWNLOAD_URL, content=tar_data)

            downloader.download(target_folder)

        assert os.path.isfile(os.path.join(target_folder, 'cv-valid-dev.csv'))
        assert os.path.isdir(os.path.join(target_folder, 'cv-valid-dev'))
        assert os.path.isfile(os.path.join(target_folder, 'cv-valid-train.csv'))
        assert os.path.isdir(os.path.join(target_folder, 'cv-valid-train'))


class TestMusanReader:

    def test_load_tracks(self, reader, sample_path):
        ds = reader.load(sample_path)

        fma = os.path.join(sample_path, 'music', 'fma')
        free_sound = os.path.join(sample_path, 'noise', 'free-sound')
        librivox = os.path.join(sample_path, 'speech', 'librivox')

        assert ds.num_tracks == 5

        assert ds.tracks['music-fma-0000'].idx == 'music-fma-0000'
        assert ds.tracks['music-fma-0000'].path == os.path.join(fma, 'music-fma-0000.wav')

        assert ds.tracks['noise-free-sound-0000'].idx == 'noise-free-sound-0000'
        assert ds.tracks['noise-free-sound-0000'].path == os.path.join(free_sound, 'noise-free-sound-0000.wav')
        assert ds.tracks['noise-free-sound-0001'].idx == 'noise-free-sound-0001'
        assert ds.tracks['noise-free-sound-0001'].path == os.path.join(free_sound, 'noise-free-sound-0001.wav')

        assert ds.tracks['speech-librivox-0000'].idx == 'speech-librivox-0000'
        assert ds.tracks['speech-librivox-0000'].path == os.path.join(librivox, 'speech-librivox-0000.wav')
        assert ds.tracks['speech-librivox-0001'].idx == 'speech-librivox-0001'
        assert ds.tracks['speech-librivox-0001'].path == os.path.join(librivox, 'speech-librivox-0001.wav')

    def test_load_issuers(self, reader, sample_path):
        ds = reader.load(sample_path)

        assert ds.num_issuers == 3

        assert 'speech-librivox-0000' in ds.issuers.keys()
        assert type(ds.issuers['speech-librivox-0000']) == issuers.Speaker
        assert ds.issuers['speech-librivox-0000'].idx == 'speech-librivox-0000'
        assert ds.issuers['speech-librivox-0000'].gender == issuers.Gender.MALE

        assert 'speech-librivox-0001' in ds.issuers.keys()
        assert type(ds.issuers['speech-librivox-0001']) == issuers.Speaker
        assert ds.issuers['speech-librivox-0001'].idx == 'speech-librivox-0001'
        assert ds.issuers['speech-librivox-0001'].gender == issuers.Gender.FEMALE

        assert 'Quiet_Music_for_Tiny_Robots' in ds.issuers.keys()
        assert type(ds.issuers['Quiet_Music_for_Tiny_Robots']) == issuers.Artist
        assert ds.issuers['Quiet_Music_for_Tiny_Robots'].idx == 'Quiet_Music_for_Tiny_Robots'
        assert ds.issuers['Quiet_Music_for_Tiny_Robots'].name == 'Quiet_Music_for_Tiny_Robots'

    def test_load_utterances(self, reader, sample_path):
        ds = reader.load(sample_path)

        assert ds.num_utterances == 5

        assert ds.utterances['music-fma-0000'].idx == 'music-fma-0000'
        assert ds.utterances['music-fma-0000'].track.idx == 'music-fma-0000'
        assert ds.utterances['music-fma-0000'].issuer.idx == 'Quiet_Music_for_Tiny_Robots'
        assert ds.utterances['music-fma-0000'].start == 0
        assert ds.utterances['music-fma-0000'].end == -1

        assert ds.utterances['noise-free-sound-0000'].idx == 'noise-free-sound-0000'
        assert ds.utterances['noise-free-sound-0000'].track.idx == 'noise-free-sound-0000'
        assert ds.utterances['noise-free-sound-0000'].issuer is None
        assert ds.utterances['noise-free-sound-0000'].start == 0
        assert ds.utterances['noise-free-sound-0000'].end == -1

        assert ds.utterances['noise-free-sound-0001'].idx == 'noise-free-sound-0001'
        assert ds.utterances['noise-free-sound-0001'].track.idx == 'noise-free-sound-0001'
        assert ds.utterances['noise-free-sound-0001'].issuer is None
        assert ds.utterances['noise-free-sound-0001'].start == 0
        assert ds.utterances['noise-free-sound-0001'].end == -1

        assert ds.utterances['speech-librivox-0000'].idx == 'speech-librivox-0000'
        assert ds.utterances['speech-librivox-0000'].track.idx == 'speech-librivox-0000'
        assert ds.utterances['speech-librivox-0000'].issuer.idx == 'speech-librivox-0000'
        assert ds.utterances['speech-librivox-0000'].start == 0
        assert ds.utterances['speech-librivox-0000'].end == -1

        assert ds.utterances['speech-librivox-0001'].idx == 'speech-librivox-0001'
        assert ds.utterances['speech-librivox-0001'].track.idx == 'speech-librivox-0001'
        assert ds.utterances['speech-librivox-0001'].issuer.idx == 'speech-librivox-0001'
        assert ds.utterances['speech-librivox-0001'].start == 0
        assert ds.utterances['speech-librivox-0001'].end == -1

    def test_load_label_lists(self, reader, sample_path):
        ds = reader.load(sample_path)

        utt_1 = ds.utterances['music-fma-0000']
        utt_2 = ds.utterances['noise-free-sound-0000']
        utt_3 = ds.utterances['noise-free-sound-0001']

        assert corpus.LL_DOMAIN in utt_1.label_lists.keys()
        assert corpus.LL_DOMAIN in utt_2.label_lists.keys()
        assert corpus.LL_DOMAIN in utt_3.label_lists.keys()

        assert len(utt_1.label_lists[corpus.LL_DOMAIN].labels) == 1
        assert len(utt_2.label_lists[corpus.LL_DOMAIN].labels) == 1
        assert len(utt_3.label_lists[corpus.LL_DOMAIN].labels) == 1

        assert utt_1.label_lists[corpus.LL_DOMAIN].labels[0].value == 'music'
        assert utt_2.label_lists[corpus.LL_DOMAIN].labels[0].value == 'noise'
        assert utt_3.label_lists[corpus.LL_DOMAIN].labels[0].value == 'noise'
