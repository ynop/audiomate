import os
import unittest

from pingu.corpus import io
from tests import resources


class MusanCorpusLoaderTest(unittest.TestCase):
    def setUp(self):
        self.loader = io.MusanLoader()
        self.test_path = resources.sample_musan_ds_path()

    def test_load_files(self):
        ds = self.loader.load(self.test_path)

        self.assertEqual(5, ds.num_files)

        self.assertEqual('music-fma-0000', ds.files['music-fma-0000'].idx)
        self.assertEqual(os.path.join(self.test_path, 'music', 'fma', 'music-fma-0000.wav'),
                         ds.files['music-fma-0000'].path)

        self.assertEqual('noise-free-sound-0000', ds.files['noise-free-sound-0000'].idx)
        self.assertEqual(os.path.join(self.test_path, 'noise', 'free-sound', 'noise-free-sound-0000.wav'),
                         ds.files['noise-free-sound-0000'].path)

        self.assertEqual('noise-free-sound-0001', ds.files['noise-free-sound-0001'].idx)
        self.assertEqual(os.path.join(self.test_path, 'noise', 'free-sound', 'noise-free-sound-0001.wav'),
                         ds.files['noise-free-sound-0001'].path)

        self.assertEqual('speech-librivox-0000', ds.files['speech-librivox-0000'].idx)
        self.assertEqual(os.path.join(self.test_path, 'speech', 'librivox', 'speech-librivox-0000.wav'),
                         ds.files['speech-librivox-0000'].path)

        self.assertEqual('speech-librivox-0001', ds.files['speech-librivox-0001'].idx)
        self.assertEqual(os.path.join(self.test_path, 'speech', 'librivox', 'speech-librivox-0001.wav'),
                         ds.files['speech-librivox-0001'].path)

    def test_load_issuers(self):
        ds = self.loader.load(self.test_path)

        self.assertEqual(3, ds.num_issuers)

        self.assertIn('speech-librivox-0000', ds.issuers.keys())
        self.assertEqual('speech-librivox-0000', ds.issuers['speech-librivox-0000'].idx)
        self.assertEqual('m', ds.issuers['speech-librivox-0000'].info['gender'])
        self.assertEqual('english', ds.issuers['speech-librivox-0000'].info['language'])

        self.assertIn('speech-librivox-0001', ds.issuers.keys())
        self.assertEqual('speech-librivox-0001', ds.issuers['speech-librivox-0001'].idx)
        self.assertEqual('f', ds.issuers['speech-librivox-0001'].info['gender'])
        self.assertEqual('french', ds.issuers['speech-librivox-0001'].info['language'])

        self.assertIn('Quiet_Music_for_Tiny_Robots', ds.issuers.keys())
        self.assertEqual('Quiet_Music_for_Tiny_Robots', ds.issuers['Quiet_Music_for_Tiny_Robots'].idx)
        self.assertIsNone(ds.issuers['Quiet_Music_for_Tiny_Robots'].info)

    def test_load_utterances(self):
        ds = self.loader.load(self.test_path)

        self.assertEqual(5, ds.num_utterances)

        self.assertEqual('music-fma-0000', ds.utterances['music-fma-0000'].idx)
        self.assertEqual('music-fma-0000', ds.utterances['music-fma-0000'].file_idx)
        self.assertEqual('Quiet_Music_for_Tiny_Robots', ds.utterances['music-fma-0000'].issuer_idx)
        self.assertEqual(0, ds.utterances['music-fma-0000'].start)
        self.assertEqual(-1, ds.utterances['music-fma-0000'].end)

        self.assertEqual('noise-free-sound-0000', ds.utterances['noise-free-sound-0000'].idx)
        self.assertEqual('noise-free-sound-0000', ds.utterances['noise-free-sound-0000'].file_idx)
        self.assertIsNone(ds.utterances['noise-free-sound-0000'].issuer_idx)
        self.assertEqual(0, ds.utterances['noise-free-sound-0000'].start)
        self.assertEqual(-1, ds.utterances['noise-free-sound-0000'].end)

        self.assertEqual('noise-free-sound-0001', ds.utterances['noise-free-sound-0001'].idx)
        self.assertEqual('noise-free-sound-0001', ds.utterances['noise-free-sound-0001'].file_idx)
        self.assertIsNone(ds.utterances['noise-free-sound-0001'].issuer_idx)
        self.assertEqual(0, ds.utterances['noise-free-sound-0001'].start)
        self.assertEqual(-1, ds.utterances['noise-free-sound-0001'].end)

        self.assertEqual('speech-librivox-0000', ds.utterances['speech-librivox-0000'].idx)
        self.assertEqual('speech-librivox-0000', ds.utterances['speech-librivox-0000'].file_idx)
        self.assertEqual('speech-librivox-0000', ds.utterances['speech-librivox-0000'].issuer_idx)
        self.assertEqual(0, ds.utterances['speech-librivox-0000'].start)
        self.assertEqual(-1, ds.utterances['speech-librivox-0000'].end)

        self.assertEqual('speech-librivox-0001', ds.utterances['speech-librivox-0001'].idx)
        self.assertEqual('speech-librivox-0001', ds.utterances['speech-librivox-0001'].file_idx)
        self.assertEqual('speech-librivox-0001', ds.utterances['speech-librivox-0001'].issuer_idx)
        self.assertEqual(0, ds.utterances['speech-librivox-0001'].start)
        self.assertEqual(-1, ds.utterances['speech-librivox-0001'].end)

    def test_load_label_lists(self):
        ds = self.loader.load(self.test_path)

        self.assertIn('music', ds.label_lists.keys())
        self.assertIn('noise', ds.label_lists.keys())

        self.assertEqual(1, len(ds.label_lists['music'].keys()))
        self.assertEqual(2, len(ds.label_lists['noise'].keys()))

        self.assertIn('music-fma-0000', ds.label_lists['music'].keys())
        self.assertIn('noise-free-sound-0000', ds.label_lists['noise'].keys())
        self.assertIn('noise-free-sound-0001', ds.label_lists['noise'].keys())

        self.assertEqual(1, len(ds.label_lists['music']['music-fma-0000'].labels))
        self.assertEqual(1, len(ds.label_lists['noise']['noise-free-sound-0000'].labels))
        self.assertEqual(1, len(ds.label_lists['noise']['noise-free-sound-0001'].labels))

        self.assertEqual('music', ds.label_lists['music']['music-fma-0000'].labels[0].value)
        self.assertEqual('noise', ds.label_lists['noise']['noise-free-sound-0000'].labels[0].value)
        self.assertEqual('noise', ds.label_lists['noise']['noise-free-sound-0001'].labels[0].value)
