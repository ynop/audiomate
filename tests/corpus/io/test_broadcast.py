import os

from audiomate import issuers
from audiomate.corpus import io

from tests import resources

import pytest


@pytest.fixture
def reader():
    return io.BroadcastReader()


@pytest.fixture
def sample_path():
    return resources.sample_corpus_path('broadcast')


class TestBroadcastReader:

    def test_load_tracks(self, reader, sample_path):
        ds = reader.load(sample_path)

        assert ds.num_tracks == 4
        assert ds.tracks['file-1'].idx == 'file-1'
        assert ds.tracks['file-1'].path == os.path.join(sample_path, 'files', 'a', 'wav_1.wav')
        assert ds.tracks['file-2'].idx == 'file-2'
        assert ds.tracks['file-2'].path == os.path.join(sample_path, 'files', 'b', 'wav_2.wav')
        assert ds.tracks['file-3'].idx == 'file-3'
        assert ds.tracks['file-3'].path == os.path.join(sample_path, 'files', 'c', 'wav_3.wav')
        assert ds.tracks['file-4'].idx == 'file-4'
        assert ds.tracks['file-4'].path == os.path.join(sample_path, 'files', 'd', 'wav_4.wav')

    def test_load_issuers(self, reader, sample_path):
        ds = reader.load(sample_path)

        assert ds.num_issuers == 3

        assert ds.issuers['speaker-1'].idx == 'speaker-1'
        assert len(ds.issuers['speaker-1'].info) == 0
        assert type(ds.issuers['speaker-1']) == issuers.Speaker
        assert ds.issuers['speaker-1'].gender == issuers.Gender.FEMALE
        assert ds.issuers['speaker-1'].age_group == issuers.AgeGroup.CHILD
        assert ds.issuers['speaker-1'].native_language == 'eng'

        assert ds.issuers['speaker-2'].idx == 'speaker-2'
        assert len(ds.issuers['speaker-2'].info) == 0
        assert type(ds.issuers['speaker-2']) == issuers.Artist
        assert ds.issuers['speaker-2'].name is None

        assert ds.issuers['speaker-3'].idx == 'speaker-3'
        assert len(ds.issuers['speaker-3'].info) == 0

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
        assert ds.utterances['utt-3'].end == 100

        assert ds.utterances['utt-4'].idx == 'utt-4'
        assert ds.utterances['utt-4'].track.idx == 'file-3'
        assert ds.utterances['utt-4'].issuer.idx == 'speaker-2'
        assert ds.utterances['utt-4'].start == 100
        assert ds.utterances['utt-4'].end == 150

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

        assert 'music' in utt_1.label_lists.keys()
        assert 'jingles' in utt_1.label_lists.keys()
        assert 'default' in utt_2.label_lists.keys()
        assert 'default' in utt_3.label_lists.keys()
        assert 'default' in utt_4.label_lists.keys()
        assert 'default' in utt_5.label_lists.keys()

        assert len(utt_1.label_lists['jingles'].labels) == 2
        assert len(utt_1.label_lists['music'].labels) == 2
        assert utt_1.label_lists['jingles'].labels[1].value == 'velo'

        assert utt_1.label_lists['jingles'].labels[1].start == 80
        assert utt_1.label_lists['jingles'].labels[1].end == 82.4

    def test_load_label_meta(self, reader, sample_path):
        ds = reader.load(sample_path)

        utt_1 = ds.utterances['utt-1']

        assert len(utt_1.label_lists['jingles'].labels[0].meta) == 0

        assert len(utt_1.label_lists['jingles'].labels[1].meta) == 3
        assert utt_1.label_lists['jingles'].labels[1].meta['lang'] == 'de'
        assert utt_1.label_lists['jingles'].labels[1].meta['prio'] == 4
        assert utt_1.label_lists['jingles'].labels[1].meta['unique']
