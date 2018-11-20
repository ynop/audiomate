import os

import pytest
import requests_mock

from audiomate import corpus
from audiomate.corpus import io
from audiomate.corpus.io import gtzan
from tests import resources


@pytest.fixture
def reader():
    return io.GtzanReader()


@pytest.fixture
def data_path():
    return resources.sample_corpus_path('gtzan')


@pytest.fixture()
def tar_data():
    with open(resources.get_resource_path(['sample_files', 'cv_corpus_v1.tar.gz']), 'rb') as f:
        return f.read()


class TestGtzanDownloader:

    def test_download(self, tar_data, tmpdir):
        target_folder = tmpdir.strpath
        downloader = io.GtzanDownloader()

        with requests_mock.Mocker() as mock:
            mock.get(gtzan.DOWNLOAD_URL, content=tar_data)

            downloader.download(target_folder)

        assert os.path.isfile(os.path.join(target_folder, 'cv-valid-dev.csv'))
        assert os.path.isdir(os.path.join(target_folder, 'cv-valid-dev'))
        assert os.path.isfile(os.path.join(target_folder, 'cv-valid-train.csv'))
        assert os.path.isdir(os.path.join(target_folder, 'cv-valid-train'))


class TestGtzanReader:

    def test_load_tracks(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_tracks == 4

        assert ds.tracks['bagpipe'].idx == 'bagpipe'
        assert ds.tracks['bagpipe'].path == os.path.join(data_path, 'music_wav', 'bagpipe.wav')

        assert ds.tracks['ballad'].idx == 'ballad'
        assert ds.tracks['ballad'].path == os.path.join(data_path, 'music_wav', 'ballad.wav')

        assert ds.tracks['acomic'].idx == 'acomic'
        assert ds.tracks['acomic'].path == os.path.join(data_path, 'speech_wav', 'acomic.wav')

        assert ds.tracks['acomic2'].idx == 'acomic2'
        assert ds.tracks['acomic2'].path == os.path.join(data_path, 'speech_wav', 'acomic2.wav')

    def test_load_issuers(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_issuers == 0

    def test_load_utterances(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_utterances == 4

        assert ds.utterances['bagpipe'].idx == 'bagpipe'
        assert ds.utterances['bagpipe'].track.idx == 'bagpipe'
        assert ds.utterances['bagpipe'].issuer is None
        assert ds.utterances['bagpipe'].start == 0
        assert ds.utterances['bagpipe'].end == -1

        assert ds.utterances['ballad'].idx == 'ballad'
        assert ds.utterances['ballad'].track.idx == 'ballad'
        assert ds.utterances['ballad'].issuer is None
        assert ds.utterances['ballad'].start == 0
        assert ds.utterances['ballad'].end == -1

        assert ds.utterances['acomic'].idx == 'acomic'
        assert ds.utterances['acomic'].track.idx == 'acomic'
        assert ds.utterances['acomic'].issuer is None
        assert ds.utterances['acomic'].start == 0
        assert ds.utterances['acomic'].end == -1

        assert ds.utterances['acomic2'].idx == 'acomic2'
        assert ds.utterances['acomic2'].track.idx == 'acomic2'
        assert ds.utterances['acomic2'].issuer is None
        assert ds.utterances['acomic2'].start == 0
        assert ds.utterances['acomic2'].end == -1

    def test_load_label_lists(self, reader, data_path):
        ds = reader.load(data_path)

        utt_1 = ds.utterances['bagpipe']
        utt_2 = ds.utterances['ballad']
        utt_3 = ds.utterances['acomic']
        utt_4 = ds.utterances['acomic2']

        assert corpus.LL_DOMAIN in utt_1.label_lists.keys()
        assert corpus.LL_DOMAIN in utt_2.label_lists.keys()
        assert corpus.LL_DOMAIN in utt_3.label_lists.keys()
        assert corpus.LL_DOMAIN in utt_4.label_lists.keys()

        assert len(utt_1.label_lists[corpus.LL_DOMAIN].labels) == 1
        assert len(utt_2.label_lists[corpus.LL_DOMAIN].labels) == 1
        assert len(utt_3.label_lists[corpus.LL_DOMAIN].labels) == 1
        assert len(utt_4.label_lists[corpus.LL_DOMAIN].labels) == 1

        assert utt_1.label_lists[corpus.LL_DOMAIN].labels[0].value == corpus.LL_DOMAIN_MUSIC
        assert utt_2.label_lists[corpus.LL_DOMAIN].labels[0].value == corpus.LL_DOMAIN_MUSIC
        assert utt_3.label_lists[corpus.LL_DOMAIN].labels[0].value == corpus.LL_DOMAIN_SPEECH
        assert utt_3.label_lists[corpus.LL_DOMAIN].labels[0].value == corpus.LL_DOMAIN_SPEECH
