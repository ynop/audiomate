import os
import unittest

from audiomate.corpus import io
from audiomate.corpus import assets
from tests import resources


class BroadcastReaderTest(unittest.TestCase):
    def setUp(self):
        self.reader = io.BroadcastReader()
        self.test_path = resources.sample_corpus_path('broadcast')

    def test_load_files(self):
        ds = self.reader.load(self.test_path)

        assert ds.num_files == 4
        assert ds.files['file-1'].idx == 'file-1'
        assert ds.files['file-1'].path == os.path.join(self.test_path, 'files', 'a', 'wav_1.wav')
        assert ds.files['file-2'].idx == 'file-2'
        assert ds.files['file-2'].path == os.path.join(self.test_path, 'files', 'b', 'wav_2.wav')
        assert ds.files['file-3'].idx == 'file-3'
        assert ds.files['file-3'].path == os.path.join(self.test_path, 'files', 'c', 'wav_3.wav')
        assert ds.files['file-4'].idx == 'file-4'
        assert ds.files['file-4'].path == os.path.join(self.test_path, 'files', 'd', 'wav_4.wav')

    def test_load_issuers(self):
        ds = self.reader.load(self.test_path)

        assert ds.num_issuers == 3

        assert ds.issuers['speaker-1'].idx == 'speaker-1'
        assert len(ds.issuers['speaker-1'].info) == 0
        assert type(ds.issuers['speaker-1']) == assets.Speaker
        assert ds.issuers['speaker-1'].gender == assets.Gender.FEMALE
        assert ds.issuers['speaker-1'].age_group == assets.AgeGroup.CHILD
        assert ds.issuers['speaker-1'].native_language == 'eng'

        assert ds.issuers['speaker-2'].idx == 'speaker-2'
        assert len(ds.issuers['speaker-2'].info) == 0
        assert type(ds.issuers['speaker-2']) == assets.Artist
        assert ds.issuers['speaker-2'].name is None

        assert ds.issuers['speaker-3'].idx == 'speaker-3'
        assert len(ds.issuers['speaker-3'].info) == 0

    def test_load_utterances(self):
        ds = self.reader.load(self.test_path)

        assert ds.num_utterances == 5

        assert ds.utterances['utt-1'].idx == 'utt-1'
        assert ds.utterances['utt-1'].file.idx == 'file-1'
        assert ds.utterances['utt-1'].issuer.idx == 'speaker-1'
        assert ds.utterances['utt-1'].start == 0
        assert ds.utterances['utt-1'].end == -1

        assert ds.utterances['utt-2'].idx == 'utt-2'
        assert ds.utterances['utt-2'].file.idx == 'file-2'
        assert ds.utterances['utt-2'].issuer.idx == 'speaker-1'
        assert ds.utterances['utt-2'].start == 0
        assert ds.utterances['utt-2'].end == -1

        assert ds.utterances['utt-3'].idx == 'utt-3'
        assert ds.utterances['utt-3'].file.idx == 'file-3'
        assert ds.utterances['utt-3'].issuer.idx == 'speaker-2'
        assert ds.utterances['utt-3'].start == 0
        assert ds.utterances['utt-3'].end == 100

        assert ds.utterances['utt-4'].idx == 'utt-4'
        assert ds.utterances['utt-4'].file.idx == 'file-3'
        assert ds.utterances['utt-4'].issuer.idx == 'speaker-2'
        assert ds.utterances['utt-4'].start == 100
        assert ds.utterances['utt-4'].end == 150

        assert ds.utterances['utt-5'].idx == 'utt-5'
        assert ds.utterances['utt-5'].file.idx == 'file-4'
        assert ds.utterances['utt-5'].issuer.idx == 'speaker-3'
        assert ds.utterances['utt-5'].start == 0
        assert ds.utterances['utt-5'].end == -1

    def test_load_label_lists(self):
        ds = self.reader.load(self.test_path)

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

    def test_load_label_meta(self):
        ds = self.reader.load(self.test_path)

        utt_1 = ds.utterances['utt-1']

        assert len(utt_1.label_lists['jingles'].labels[0].meta) == 0

        assert len(utt_1.label_lists['jingles'].labels[1].meta) == 3
        assert utt_1.label_lists['jingles'].labels[1].meta['lang'] == 'de'
        assert utt_1.label_lists['jingles'].labels[1].meta['prio'] == 4
        assert utt_1.label_lists['jingles'].labels[1].meta['unique']
