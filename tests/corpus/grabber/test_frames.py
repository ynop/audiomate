import os
import shutil
import tempfile
import unittest

import numpy as np

import pingu
from pingu.corpus import assets
from pingu.corpus import grabber


class FrameClassificationGrabberGrabberTest(unittest.TestCase):
    def setUp(self):
        self.temp_path = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_path, ignore_errors=True)

    def test_items(self):
        ds = pingu.Corpus()

        ds.new_file('not/important', 'a')
        ds.new_file('who/cares', 'b')

        utt_a1 = ds.new_utterance('a1', 'a', start=0.0, end=2.0)
        utt_a2 = ds.new_utterance('a2', 'a', start=2.0, end=3.75)
        utt_b1 = ds.new_utterance('b1', 'b', start=0.0, end=1.02)

        utt_a1.set_label_list(assets.LabelList(idx='dudelida', labels=[
            assets.Label('chi', 0.0, 1.0),
            assets.Label('cha', 1.0, 1.5),
            assets.Label('wottinid', 1.5, 2.0)
        ]))

        utt_a2.set_label_list(assets.LabelList(idx='dudelida', labels=[
            assets.Label('cha', 0.2, 1.2)
        ]))

        utt_b1.set_label_list(assets.LabelList(idx='dudelida', labels=[
            assets.Label('chi', 0.3, 0.9)
        ]))

        feats = assets.FeatureContainer(os.path.join(self.temp_path, 'feats'))
        feats.open()

        feats.frame_size = 4
        feats.hop_size = 2
        feats.sampling_rate = 8

        feats.set('a1', np.array([
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 0],
        ]))

        feats.set('a2', np.array([
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 0, 0]
        ]))

        feats.set('b1', np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 0]
        ]))

        gr = grabber.FrameClassificationGrabber(ds, feats, label_list_idx='dudelida', label_values=['chi', 'cha'])

        self.assertEqual(14, len(gr))

        # a1
        frame, label_vec = gr[0]
        self.assertTrue(np.allclose(np.array([0, 1, 0, 1], np.float), frame))
        self.assertTrue(np.allclose([1, 0], label_vec))

        frame, label_vec = gr[1]
        self.assertTrue(np.allclose([0, 1, 1, 0], frame))
        self.assertTrue(np.allclose([1, 0], label_vec))

        frame, label_vec = gr[2]
        self.assertTrue(np.allclose([1, 0, 0, 0], frame))
        self.assertTrue(np.allclose([1, 0], label_vec))

        frame, label_vec = gr[3]
        self.assertTrue(np.allclose([0, 0, 1, 1], frame))
        self.assertTrue(np.allclose([1, 0], label_vec))

        frame, label_vec = gr[4]
        self.assertTrue(np.allclose([1, 1, 1, 0], frame))
        self.assertTrue(np.allclose([0, 1], label_vec))

        frame, label_vec = gr[5]
        self.assertTrue(np.allclose([1, 0, 1, 0], frame))
        self.assertTrue(np.allclose([0, 1], label_vec))

        # a2
        frame, label_vec = gr[6]
        self.assertTrue(np.allclose([1, 0, 1, 0], frame))
        self.assertTrue(np.allclose([0, 1], label_vec))

        frame, label_vec = gr[7]
        self.assertTrue(np.allclose([1, 0, 0, 1], frame))
        self.assertTrue(np.allclose([0, 1], label_vec))

        frame, label_vec = gr[8]
        self.assertTrue(np.allclose([0, 1, 0, 0], frame))
        self.assertTrue(np.allclose([0, 1], label_vec))

        frame, label_vec = gr[9]
        self.assertTrue(np.allclose([0, 0, 1, 0], frame))
        self.assertTrue(np.allclose([0, 1], label_vec))

        frame, label_vec = gr[10]
        self.assertTrue(np.allclose([1, 0, 0, 1], frame))
        self.assertTrue(np.allclose([0, 1], label_vec))

        # b1
        frame, label_vec = gr[11]
        self.assertTrue(np.allclose([0, 0, 1, 0], frame))
        self.assertTrue(np.allclose([1, 0], label_vec))

        frame, label_vec = gr[12]
        self.assertTrue(np.allclose([1, 0, 0, 0], frame))
        self.assertTrue(np.allclose([1, 0], label_vec))

        frame, label_vec = gr[13]
        self.assertTrue(np.allclose([0, 1, 1, 0], frame))
        self.assertTrue(np.allclose([1, 0], label_vec))

        feats.close()

    def test_items_one_label_per_utterance(self):
        ds = pingu.Corpus()

        ds.new_file('not/important', 'a')

        utt_a1 = ds.new_utterance('a1', 'a', start=0.0, end=2.0)
        utt_a2 = ds.new_utterance('a2', 'a', start=2.0, end=3.75)

        utt_a1.set_label_list(assets.LabelList(idx='dudelida', labels=[
            assets.Label('chi')
        ]))

        utt_a2.set_label_list(assets.LabelList(idx='dudelida', labels=[
            assets.Label('cha')
        ]))

        feats = assets.FeatureContainer(os.path.join(self.temp_path, 'feats'))
        feats.open()

        feats.frame_size = 4
        feats.hop_size = 2
        feats.sampling_rate = 8

        feats.set('a1', np.array([
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 0],
        ]))

        feats.set('a2', np.array([
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 0, 0]
        ]))

        gr = grabber.FrameClassificationGrabber(ds, feats, label_list_idx='dudelida')

        self.assertEqual(15, len(gr))

        # a1
        frame, label_vec = gr[0]
        self.assertTrue(np.allclose([0, 1, 0, 1], frame))
        self.assertTrue(np.allclose([0, 1], label_vec))

        frame, label_vec = gr[1]
        self.assertTrue(np.allclose([0, 1, 1, 0], frame))
        self.assertTrue(np.allclose([0, 1], label_vec))

        frame, label_vec = gr[2]
        self.assertTrue(np.allclose([1, 0, 0, 0], frame))
        self.assertTrue(np.allclose([0, 1], label_vec))

        frame, label_vec = gr[3]
        self.assertTrue(np.allclose([0, 0, 1, 1], frame))
        self.assertTrue(np.allclose([0, 1], label_vec))

        frame, label_vec = gr[4]
        self.assertTrue(np.allclose([1, 1, 1, 0], frame))
        self.assertTrue(np.allclose([0, 1], label_vec))

        frame, label_vec = gr[5]
        self.assertTrue(np.allclose([1, 0, 1, 0], frame))
        self.assertTrue(np.allclose([0, 1], label_vec))

        frame, label_vec = gr[6]
        self.assertTrue(np.allclose([1, 0, 1, 0], frame))
        self.assertTrue(np.allclose([0, 1], label_vec))

        frame, label_vec = gr[7]
        self.assertTrue(np.allclose([1, 0, 0, 0], frame))
        self.assertTrue(np.allclose([0, 1], label_vec))

        # a2
        frame, label_vec = gr[8]
        self.assertTrue(np.allclose([1, 0, 1, 0], frame))
        self.assertTrue(np.allclose([1, 0], label_vec))

        frame, label_vec = gr[9]
        self.assertTrue(np.allclose([1, 0, 0, 1], frame))
        self.assertTrue(np.allclose([1, 0], label_vec))

        frame, label_vec = gr[10]
        self.assertTrue(np.allclose([0, 1, 0, 0], frame))
        self.assertTrue(np.allclose([1, 0], label_vec))

        frame, label_vec = gr[11]
        self.assertTrue(np.allclose([0, 0, 1, 0], frame))
        self.assertTrue(np.allclose([1, 0], label_vec))

        frame, label_vec = gr[12]
        self.assertTrue(np.allclose([1, 0, 0, 1], frame))
        self.assertTrue(np.allclose([1, 0], label_vec))

        frame, label_vec = gr[13]
        self.assertTrue(np.allclose([0, 1, 0, 0], frame))
        self.assertTrue(np.allclose([1, 0], label_vec))

        frame, label_vec = gr[14]
        self.assertTrue(np.allclose([0, 0, 0, 0], frame))
        self.assertTrue(np.allclose([1, 0], label_vec))

        feats.close()
