import os
import shutil
import tempfile
import unittest

from pingu.corpus import io
from tests import resources


class DefaultCorpusLoaderTest(unittest.TestCase):
    def setUp(self):
        self.loader = io.DefaultLoader()
        self.test_path = resources.sample_default_ds_path()

    def tearDown(self):
        pass

    def test_load_files(self):
        ds = self.loader.load(self.test_path)

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
        ds = self.loader.load(self.test_path)

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
        ds = self.loader.load(self.test_path)

        self.assertIn('text', ds.label_lists.keys())
        self.assertIn('utt-1', ds.label_lists['text'].keys())
        self.assertIn('utt-3', ds.label_lists['raw_text'].keys())

        self.assertEqual(3, len(ds.label_lists['text']['utt-4'].labels))
        self.assertEqual('are', ds.label_lists['text']['utt-4'].labels[1].value)

        self.assertEqual(3.5, ds.label_lists['text']['utt-4'].labels[2].start)
        self.assertEqual(4.2, ds.label_lists['text']['utt-4'].labels[2].end)

    def test_load_features(self):
        ds = self.loader.load(self.test_path)

        self.assertEqual(os.path.join(self.test_path, 'features', 'mfcc'),
                         ds.feature_containers['mfcc'].path)
        self.assertEqual(os.path.join(self.test_path, 'features', 'fbank'),
                         ds.feature_containers['fbank'].path)

    def test_save(self):
        ds = resources.create_dataset()
        path = tempfile.mkdtemp()
        self.loader.save(ds, path)

        self.assertIn('files.txt', os.listdir(path))
        self.assertIn('utterances.txt', os.listdir(path))
        self.assertIn('utt_issuers.txt', os.listdir(path))
        self.assertIn('labels_default.txt', os.listdir(path))

        shutil.rmtree(path, ignore_errors=True)
