import os
import unittest

from pingu.corpus import io
from tests import resources


class DefaultCorpusReaderTest(unittest.TestCase):
    def setUp(self):
        self.reader = io.BroadcastReader()
        self.test_path = resources.sample_broadcast_ds_path()

    def test_load_files(self):
        ds = self.reader.load(self.test_path)

        self.assertEqual(4, ds.num_files)
        self.assertEqual('file-1', ds.files['file-1'].idx)
        self.assertEqual(os.path.join(self.test_path, 'files', 'a', 'wav_1.wav'),
                         ds.files['file-1'].path)
        self.assertEqual('file-2', ds.files['file-2'].idx)
        self.assertEqual(os.path.join(self.test_path, 'files', 'b', 'wav_2.wav'),
                         ds.files['file-2'].path)
        self.assertEqual('file-3', ds.files['file-3'].idx)
        self.assertEqual(os.path.join(self.test_path, 'files', 'c', 'wav_3.wav'),
                         ds.files['file-3'].path)
        self.assertEqual('file-4', ds.files['file-4'].idx)
        self.assertEqual(os.path.join(self.test_path, 'files', 'd', 'wav_4.wav'),
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
        self.assertEqual(100, ds.utterances['utt-3'].end)

        self.assertEqual('utt-4', ds.utterances['utt-4'].idx)
        self.assertEqual('file-3', ds.utterances['utt-4'].file_idx)
        self.assertEqual('speaker-2', ds.utterances['utt-4'].issuer_idx)
        self.assertEqual(100, ds.utterances['utt-4'].start)
        self.assertEqual(150, ds.utterances['utt-4'].end)

        self.assertEqual('utt-5', ds.utterances['utt-5'].idx)
        self.assertEqual('file-4', ds.utterances['utt-5'].file_idx)
        self.assertEqual('speaker-3', ds.utterances['utt-5'].issuer_idx)
        self.assertEqual(0, ds.utterances['utt-5'].start)
        self.assertEqual(-1, ds.utterances['utt-5'].end)

    def test_load_label_lists(self):
        ds = self.reader.load(self.test_path)

        self.assertIn('default', ds.label_lists.keys())
        self.assertIn('music', ds.label_lists.keys())
        self.assertIn('jingles', ds.label_lists.keys())

        self.assertIn('utt-1', ds.label_lists['jingles'].keys())
        self.assertIn('utt-1', ds.label_lists['music'].keys())
        self.assertIn('utt-2', ds.label_lists['default'].keys())
        self.assertIn('utt-3', ds.label_lists['default'].keys())
        self.assertIn('utt-4', ds.label_lists['default'].keys())
        self.assertIn('utt-5', ds.label_lists['default'].keys())

        self.assertEqual(2, len(ds.label_lists['jingles']['utt-1'].labels))
        self.assertEqual(2, len(ds.label_lists['music']['utt-1'].labels))
        self.assertEqual('velo', ds.label_lists['jingles']['utt-1'].labels[1].value)

        self.assertEqual(80, ds.label_lists['jingles']['utt-1'].labels[1].start)
        self.assertEqual(82.4, ds.label_lists['jingles']['utt-1'].labels[1].end)
