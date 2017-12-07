import os
import shutil
import tempfile
import unittest

from pingu.corpus import io
from tests import resources


class KaldiReaderTest(unittest.TestCase):
    def setUp(self):
        self.reader = io.KaldiReader()
        self.test_path = resources.sample_kaldi_ds_path()

    def test_load_files(self):
        ds = self.reader.load(self.test_path)

        self.assertEqual(4, ds.num_files)
        self.assertEqual('file-1', ds.files['file-1'].idx)
        self.assertEqual(os.path.join(self.test_path, 'files', 'wav_1.wav'),
                         ds.files['file-1'].path)
        self.assertEqual('file-2', ds.files['file-2'].idx)
        self.assertEqual(os.path.join(self.test_path, 'files', 'wav_2.wav'),
                         ds.files['file-2'].path)
        self.assertEqual('file-3', ds.files['file-3'].idx)
        self.assertEqual(os.path.join(self.test_path, 'files', 'wav_3.wav'),
                         ds.files['file-3'].path)
        self.assertEqual('file-4', ds.files['file-4'].idx)
        self.assertEqual(os.path.join(self.test_path, 'files', 'wav_4.wav'),
                         ds.files['file-4'].path)

    def test_load_utterances(self):
        ds = self.reader.load(self.test_path)

        self.assertEqual(5, ds.num_utterances)

        self.assertEqual('utt-1', ds.utterances['utt-1'].idx)
        self.assertEqual('file-1', ds.utterances['utt-1'].file_idx)
        self.assertEqual('speaker-1', ds.utterances['utt-1'].issuer_idx)
        self.assertEqual(0, ds.utterances['utt-1'].start)
        self.assertEqual(-1, ds.utterances['utt-1'].end)

        self.assertEqual('utt-2', ds.utterances['utt-2'].idx)
        self.assertEqual('file-2', ds.utterances['utt-2'].file_idx)
        self.assertEqual('speaker-1', ds.utterances['utt-2'].issuer_idx)
        self.assertEqual(0, ds.utterances['utt-2'].start)
        self.assertEqual(-1, ds.utterances['utt-2'].end)

        self.assertEqual('utt-3', ds.utterances['utt-3'].idx)
        self.assertEqual('file-3', ds.utterances['utt-3'].file_idx)
        self.assertEqual('speaker-2', ds.utterances['utt-3'].issuer_idx)
        self.assertEqual(0, ds.utterances['utt-3'].start)
        self.assertEqual(15, ds.utterances['utt-3'].end)

        self.assertEqual('utt-4', ds.utterances['utt-4'].idx)
        self.assertEqual('file-3', ds.utterances['utt-4'].file_idx)
        self.assertEqual('speaker-2', ds.utterances['utt-4'].issuer_idx)
        self.assertEqual(15, ds.utterances['utt-4'].start)
        self.assertEqual(25, ds.utterances['utt-4'].end)

        self.assertEqual('utt-5', ds.utterances['utt-5'].idx)
        self.assertEqual('file-4', ds.utterances['utt-5'].file_idx)
        self.assertEqual('speaker-3', ds.utterances['utt-5'].issuer_idx)
        self.assertEqual(0, ds.utterances['utt-5'].start)
        self.assertEqual(-1, ds.utterances['utt-5'].end)

    def test_load_label_lists(self):
        ds = self.reader.load(self.test_path)

        self.assertIn('default', ds.label_lists.keys())
        self.assertIn('utt-1', ds.label_lists['default'].keys())

        self.assertEqual(1, len(ds.label_lists['default']['utt-2'].labels))
        self.assertEqual('who are you', ds.label_lists['default']['utt-2'].labels[0].value)

        self.assertEqual(0, ds.label_lists['default']['utt-4'].labels[0].start)
        self.assertEqual(-1, ds.label_lists['default']['utt-4'].labels[0].end)


class KaldiWriterTest(unittest.TestCase):
    def setUp(self):
        self.writer = io.KaldiWriter()
        self.test_path = resources.sample_kaldi_ds_path()

    def test_save(self):
        ds = resources.create_dataset()
        path = tempfile.mkdtemp()
        self.writer.save(ds, path)

        self.assertIn('segments', os.listdir(path))
        self.assertIn('text', os.listdir(path))
        self.assertIn('utt2spk', os.listdir(path))
        self.assertIn('wav.scp', os.listdir(path))

        shutil.rmtree(path, ignore_errors=True)
