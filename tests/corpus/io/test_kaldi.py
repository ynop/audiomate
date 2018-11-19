import os

from audiomate import corpus
from audiomate import issuers
from audiomate.corpus import io

import pytest

from tests import resources


@pytest.fixture
def reader():
    return io.KaldiReader()


@pytest.fixture
def writer():
    return io.KaldiWriter()


@pytest.fixture
def sample_path():
    return resources.sample_corpus_path('kaldi')


class TestKaldiReader:

    def test_load_tracks(self, reader, sample_path):
        ds = reader.load(sample_path)

        assert ds.num_tracks == 4

        assert ds.tracks['file-1'].idx == 'file-1'
        assert ds.tracks['file-1'].path == os.path.join(sample_path, 'files', 'wav_1.wav')
        assert ds.tracks['file-2'].idx == 'file-2'
        assert ds.tracks['file-2'].path == os.path.join(sample_path, 'files', 'wav_2.wav')
        assert ds.tracks['file-3'].idx == 'file-3'
        assert ds.tracks['file-3'].path == os.path.join(sample_path, 'files', 'wav_3.wav')
        assert ds.tracks['file-4'].idx == 'file-4'
        assert ds.tracks['file-4'].path == os.path.join(sample_path, 'files', 'wav_4.wav')

    def test_load_issuers(self, reader, sample_path):
        ds = reader.load(sample_path)

        assert ds.num_issuers == 3

        assert ds.issuers['speaker-1'].idx == 'speaker-1'
        assert type(ds.issuers['speaker-1']) == issuers.Speaker
        assert ds.issuers['speaker-1'].gender == issuers.Gender.MALE
        assert ds.issuers['speaker-1'].age_group == issuers.AgeGroup.UNKNOWN
        assert ds.issuers['speaker-1'].native_language is None

        assert ds.issuers['speaker-2'].idx == 'speaker-2'
        assert type(ds.issuers['speaker-2']) == issuers.Speaker
        assert ds.issuers['speaker-2'].gender == issuers.Gender.MALE
        assert ds.issuers['speaker-2'].age_group == issuers.AgeGroup.UNKNOWN
        assert ds.issuers['speaker-2'].native_language is None

        assert ds.issuers['speaker-3'].idx == 'speaker-3'
        assert type(ds.issuers['speaker-3']) == issuers.Speaker
        assert ds.issuers['speaker-3'].gender == issuers.Gender.FEMALE
        assert ds.issuers['speaker-3'].age_group == issuers.AgeGroup.UNKNOWN
        assert ds.issuers['speaker-3'].native_language is None

    def test_load_utterances(self, reader, sample_path):
        ds = reader.load(sample_path)

        assert ds.num_utterances == 5

        assert ds.utterances['utt-1'].idx == 'utt-1'
        assert ds.utterances['utt-1'].track.idx == 'file-1'
        assert ds.utterances['utt-1'].issuer.idx == 'speaker-1'
        assert ds.utterances['utt-1'].start == 0
        assert ds.utterances['utt-1'].end == -1

        assert ds.utterances['utt-2'].idx == 'utt-2'
        assert ds.utterances['utt-2'].track.idx == 'file-2'
        assert ds.utterances['utt-2'].issuer.idx == 'speaker-1'
        assert ds.utterances['utt-2'].start == 0
        assert ds.utterances['utt-2'].end == -1

        assert ds.utterances['utt-3'].idx == 'utt-3'
        assert ds.utterances['utt-3'].track.idx == 'file-3'
        assert ds.utterances['utt-3'].issuer.idx == 'speaker-2'
        assert ds.utterances['utt-3'].start == 0
        assert ds.utterances['utt-3'].end == 15

        assert ds.utterances['utt-4'].idx == 'utt-4'
        assert ds.utterances['utt-4'].track.idx == 'file-3'
        assert ds.utterances['utt-4'].issuer.idx == 'speaker-2'
        assert ds.utterances['utt-4'].start == 15
        assert ds.utterances['utt-4'].end == 25

        assert ds.utterances['utt-5'].idx == 'utt-5'
        assert ds.utterances['utt-5'].track.idx == 'file-4'
        assert ds.utterances['utt-5'].issuer.idx == 'speaker-3'
        assert ds.utterances['utt-5'].start == 0
        assert ds.utterances['utt-5'].end == -1

    def test_load_label_lists(self, reader, sample_path):
        ds = reader.load(sample_path)

        utt_1 = ds.utterances['utt-1']
        utt_2 = ds.utterances['utt-2']
        utt_3 = ds.utterances['utt-3']
        utt_4 = ds.utterances['utt-4']
        utt_5 = ds.utterances['utt-5']

        assert corpus.LL_WORD_TRANSCRIPT in utt_1.label_lists.keys()
        assert corpus.LL_WORD_TRANSCRIPT in utt_2.label_lists.keys()
        assert corpus.LL_WORD_TRANSCRIPT in utt_3.label_lists.keys()
        assert corpus.LL_WORD_TRANSCRIPT in utt_4.label_lists.keys()
        assert corpus.LL_WORD_TRANSCRIPT in utt_5.label_lists.keys()

        assert len(utt_4.label_lists[corpus.LL_WORD_TRANSCRIPT].labels) == 1
        assert utt_4.label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value == 'who are they'

        assert utt_4.label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].start == 0
        assert utt_4.label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].end == -1


class TestKaldiWriter:

    def test_save(self, writer, tmpdir):
        ds = resources.create_dataset()
        path = tmpdir.strpath
        writer.save(ds, path)

        assert 'segments' in os.listdir(path)
        assert 'text' in os.listdir(path)
        assert 'utt2spk' in os.listdir(path)
        assert 'spk2gender' in os.listdir(path)
        assert 'wav.scp' in os.listdir(path)
