import os
import shutil
import tempfile
import unittest

from pingu.corpus import io
from tests import resources


class DefaultReaderTest(unittest.TestCase):
    def setUp(self):
        self.reader = io.DefaultReader()
        self.test_path = resources.sample_default_ds_path()

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
        assert ds.utterances['utt-3'].end == 1.5

        assert ds.utterances['utt-4'].idx == 'utt-4'
        assert ds.utterances['utt-4'].file.idx == 'file-3'
        assert ds.utterances['utt-4'].issuer.idx == 'speaker-2'
        assert ds.utterances['utt-4'].start == 1.5
        assert ds.utterances['utt-4'].end == 2.5

        assert ds.utterances['utt-5'].idx == 'utt-5'
        assert ds.utterances['utt-5'].file.idx == 'file-4'
        assert ds.utterances['utt-5'].issuer.idx == 'speaker-3'
        assert ds.utterances['utt-5'].start == 0
        assert ds.utterances['utt-5'].end == -1

    def test_load_label_lists(self):
        ds = self.reader.load(self.test_path)

        utt_1 = ds.utterances['utt-1']
        utt_3 = ds.utterances['utt-3']
        utt_4 = ds.utterances['utt-4']

        assert 'text' in utt_1.label_lists.keys()
        assert 'raw_text' in utt_3.label_lists.keys()

        assert len(utt_4.label_lists['text'].labels) == 3
        assert utt_4.label_lists['text'].labels[1].value == 'are'

        assert utt_4.label_lists['text'].labels[2].start == 3.5
        assert utt_4.label_lists['text'].labels[2].end == 4.2

    def test_load_features(self):
        ds = self.reader.load(self.test_path)

        assert ds.feature_containers['mfcc'].path == os.path.join(self.test_path, 'features', 'mfcc')
        assert ds.feature_containers['fbank'].path == os.path.join(self.test_path, 'features', 'fbank')

    def test_load_subviews(self):
        ds = self.reader.load(self.test_path)

        assert 'train' in ds.subviews.keys()
        assert 'dev' in ds.subviews.keys()

        assert len(ds.subviews['train'].filter_criteria) == 1
        assert len(ds.subviews['dev'].filter_criteria) == 1

        assert ds.subviews['train'].filter_criteria[0].utterance_idxs == {'utt-1', 'utt-2', 'utt-3'}
        assert ds.subviews['dev'].filter_criteria[0].utterance_idxs == {'utt-4', 'utt-5'}

        assert not ds.subviews['train'].filter_criteria[0].inverse
        assert not ds.subviews['dev'].filter_criteria[0].inverse


class DefaultWriterTest(unittest.TestCase):
    def setUp(self):
        self.writer = io.DefaultWriter()
        self.test_path = resources.sample_default_ds_path()

    def test_save(self):
        ds = resources.create_dataset()
        path = tempfile.mkdtemp()
        self.writer.save(ds, path)

        assert 'files.txt' in os.listdir(path)
        assert 'utterances.txt' in os.listdir(path)
        assert 'utt_issuers.txt' in os.listdir(path)
        assert 'labels_default.txt' in os.listdir(path)

        shutil.rmtree(path, ignore_errors=True)
