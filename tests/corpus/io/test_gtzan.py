import os

import pytest

from audiomate.corpus import io
from tests import resources


@pytest.fixture
def reader():
    return io.GtzanReader()


@pytest.fixture
def data_path():
    return resources.sample_corpus_path('gtzan')


class TestGtzanReader:

    def test_load_files(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_files == 4

        assert ds.files['bagpipe'].idx == 'bagpipe'
        assert ds.files['bagpipe'].path == os.path.join(data_path, 'music_wav', 'bagpipe.wav')

        assert ds.files['ballad'].idx == 'ballad'
        assert ds.files['ballad'].path == os.path.join(data_path, 'music_wav', 'ballad.wav')

        assert ds.files['acomic'].idx == 'acomic'
        assert ds.files['acomic'].path == os.path.join(data_path, 'speech_wav', 'acomic.wav')

        assert ds.files['acomic2'].idx == 'acomic2'
        assert ds.files['acomic2'].path == os.path.join(data_path, 'speech_wav', 'acomic2.wav')

    def test_load_issuers(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_issuers == 0

    def test_load_utterances(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_utterances == 4

        assert ds.utterances['bagpipe'].idx == 'bagpipe'
        assert ds.utterances['bagpipe'].file.idx == 'bagpipe'
        assert ds.utterances['bagpipe'].issuer is None
        assert ds.utterances['bagpipe'].start == 0
        assert ds.utterances['bagpipe'].end == -1

        assert ds.utterances['ballad'].idx == 'ballad'
        assert ds.utterances['ballad'].file.idx == 'ballad'
        assert ds.utterances['ballad'].issuer is None
        assert ds.utterances['ballad'].start == 0
        assert ds.utterances['ballad'].end == -1

        assert ds.utterances['acomic'].idx == 'acomic'
        assert ds.utterances['acomic'].file.idx == 'acomic'
        assert ds.utterances['acomic'].issuer is None
        assert ds.utterances['acomic'].start == 0
        assert ds.utterances['acomic'].end == -1

        assert ds.utterances['acomic2'].idx == 'acomic2'
        assert ds.utterances['acomic2'].file.idx == 'acomic2'
        assert ds.utterances['acomic2'].issuer is None
        assert ds.utterances['acomic2'].start == 0
        assert ds.utterances['acomic2'].end == -1

    def test_load_label_lists(self, reader, data_path):
        ds = reader.load(data_path)

        utt_1 = ds.utterances['bagpipe']
        utt_2 = ds.utterances['ballad']
        utt_3 = ds.utterances['acomic']
        utt_4 = ds.utterances['acomic2']

        assert 'audio_type' in utt_1.label_lists.keys()
        assert 'audio_type' in utt_2.label_lists.keys()
        assert 'audio_type' in utt_3.label_lists.keys()
        assert 'audio_type' in utt_4.label_lists.keys()

        assert len(utt_1.label_lists['audio_type'].labels) == 1
        assert len(utt_2.label_lists['audio_type'].labels) == 1
        assert len(utt_3.label_lists['audio_type'].labels) == 1
        assert len(utt_4.label_lists['audio_type'].labels) == 1

        assert utt_1.label_lists['audio_type'].labels[0].value == 'music'
        assert utt_2.label_lists['audio_type'].labels[0].value == 'music'
        assert utt_3.label_lists['audio_type'].labels[0].value == 'speech'
        assert utt_3.label_lists['audio_type'].labels[0].value == 'speech'
