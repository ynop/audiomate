import shutil
import os
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

import pingu
from pingu.corpus import assets
from pingu.corpus import grabber


class FramedSignalGrabberTest(unittest.TestCase):
    def setUp(self):
        self.temp_path = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_path, ignore_errors=True)

    def test_items(self):
        file_a_data = np.array([0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1]).astype(np.int16)
        file_b_data = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1]).astype(np.int16)

        file_a_data *= np.iinfo(np.int16).max
        file_b_data *= np.iinfo(np.int16).max

        file_a_path = os.path.join(self.temp_path, 'a.wav')
        file_b_path = os.path.join(self.temp_path, 'b.wav')

        wavfile.write(file_a_path, 4, file_a_data)
        wavfile.write(file_b_path, 4, file_b_data)

        ds = pingu.Corpus()

        ds.new_file(file_a_path, 'a')
        ds.new_file(file_b_path, 'b')

        ds.new_utterance('a1', 'a', start=0.25, end=1.5)
        ds.new_utterance('a2', 'a', start=1.5, end=3.0)
        ds.new_utterance('b1', 'b', start=0.0, end=1.02)

        ds.new_label_list('a1', 'dudelida', labels=[
            assets.Label('chi', 0.0, 1.0),
            assets.Label('cha', 1.0, 1.25),
            assets.Label('wottinid', 1.25, 1.3)
        ])

        ds.new_label_list('a2', 'dudelida', labels=[
            assets.Label('cha', 0.2, 1.2)
        ])

        ds.new_label_list('b1', 'dudelida', labels=[
            assets.Label('chi', 0.3, 0.9)
        ])

        gr = grabber.FramedSignalGrabber(ds, label_list_idx='dudelida', frame_length=4, hop_size=2, include_labels=['chi', 'cha'])

        self.assertEqual(6, len(gr))

        # a1
        frame, label_vec = gr[0]
        self.assertTrue(np.allclose(np.array([0, 1, 0, 0], np.float), frame))
        self.assertTrue(np.allclose([0, 1], label_vec))

        frame, label_vec = gr[1]
        self.assertTrue(np.allclose([0, 0, 0, 0], frame))
        self.assertTrue(np.allclose([0, 1], label_vec))

        frame, label_vec = gr[2]
        self.assertTrue(np.allclose([0, 0, 0, 0], frame))
        self.assertTrue(np.allclose([1, 0], label_vec))

        # a2
        frame, label_vec = gr[3]
        self.assertTrue(np.allclose([1, 0, 1, 0], frame))
        self.assertTrue(np.allclose([1, 0], label_vec))

        frame, label_vec = gr[4]
        self.assertTrue(np.allclose([1, 0, 0, 0], frame))
        self.assertTrue(np.allclose([1, 0], label_vec))

        # b1
        frame, label_vec = gr[5]
        self.assertTrue(np.allclose([1, 0, 0, 0], frame))
        self.assertTrue(np.allclose([0, 1], label_vec))