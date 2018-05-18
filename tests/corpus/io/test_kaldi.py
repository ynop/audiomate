import os
import shutil
import tempfile
import unittest

from audiomate.corpus import io
from audiomate.corpus import assets
from tests import resources


class KaldiReaderTest(unittest.TestCase):
    def setUp(self):
        self.reader = io.KaldiReader()
        self.test_path = resources.sample_corpus_path('kaldi')

    def test_load_files(self):
        ds = self.reader.load(self.test_path)

        assert ds.num_files == 4

        assert ds.files['file-1'].idx == 'file-1'
        assert ds.files['file-1'].path == os.path.join(self.test_path, 'files', 'wav_1.wav')
        assert ds.files['file-2'].idx == 'file-2'
        assert ds.files['file-2'].path == os.path.join(self.test_path, 'files', 'wav_2.wav')
        assert ds.files['file-3'].idx == 'file-3'
        assert ds.files['file-3'].path == os.path.join(self.test_path, 'files', 'wav_3.wav')
        assert ds.files['file-4'].idx == 'file-4'
        assert ds.files['file-4'].path == os.path.join(self.test_path, 'files', 'wav_4.wav')

    def test_load_issuers(self):
        ds = self.reader.load(self.test_path)

        assert ds.num_issuers == 3

        assert ds.issuers['speaker-1'].idx == 'speaker-1'
        assert type(ds.issuers['speaker-1']) == assets.Speaker
        assert ds.issuers['speaker-1'].gender == assets.Gender.MALE
        assert ds.issuers['speaker-1'].age_group == assets.AgeGroup.UNKNOWN
        assert ds.issuers['speaker-1'].native_language is None

        assert ds.issuers['speaker-2'].idx == 'speaker-2'
        assert type(ds.issuers['speaker-2']) == assets.Speaker
        assert ds.issuers['speaker-2'].gender == assets.Gender.MALE
        assert ds.issuers['speaker-2'].age_group == assets.AgeGroup.UNKNOWN
        assert ds.issuers['speaker-2'].native_language is None

        assert ds.issuers['speaker-3'].idx == 'speaker-3'
        assert type(ds.issuers['speaker-3']) == assets.Speaker
        assert ds.issuers['speaker-3'].gender == assets.Gender.FEMALE
        assert ds.issuers['speaker-3'].age_group == assets.AgeGroup.UNKNOWN
        assert ds.issuers['speaker-3'].native_language is None

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
        assert ds.utterances['utt-3'].end == 15

        assert ds.utterances['utt-4'].idx == 'utt-4'
        assert ds.utterances['utt-4'].file.idx == 'file-3'
        assert ds.utterances['utt-4'].issuer.idx == 'speaker-2'
        assert ds.utterances['utt-4'].start == 15
        assert ds.utterances['utt-4'].end == 25

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

        assert 'default' in utt_1.label_lists.keys()
        assert 'default' in utt_2.label_lists.keys()
        assert 'default' in utt_3.label_lists.keys()
        assert 'default' in utt_4.label_lists.keys()
        assert 'default' in utt_5.label_lists.keys()

        assert len(utt_4.label_lists['default'].labels) == 1
        assert utt_4.label_lists['default'].labels[0].value == 'who are they'

        assert utt_4.label_lists['default'].labels[0].start == 0
        assert utt_4.label_lists['default'].labels[0].end == -1


class KaldiWriterTest(unittest.TestCase):
    def setUp(self):
        self.writer = io.KaldiWriter()

    def test_save(self):
        ds = resources.create_dataset()
        path = tempfile.mkdtemp()
        self.writer.save(ds, path)

        assert 'segments' in os.listdir(path)
        assert 'text' in os.listdir(path)
        assert 'utt2spk' in os.listdir(path)
        assert 'spk2gender' in os.listdir(path)
        assert 'wav.scp' in os.listdir(path)

        shutil.rmtree(path, ignore_errors=True)
