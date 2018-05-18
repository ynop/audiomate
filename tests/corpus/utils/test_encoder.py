import unittest

import numpy as np

from audiomate.corpus.utils import label_encoding
from audiomate.corpus import assets
from audiomate.utils import units

from tests import resources


class TestFrameOneHotEncoder(unittest.TestCase):

    def test_encode_full_utterance(self):
        ds = resources.create_multi_label_corpus()
        enc = label_encoding.FrameOneHotEncoder(['music', 'speech', 'noise'],
                                                frame_settings=units.FrameSettings(32000, 16000),
                                                sr=16000)

        actual = enc.encode(ds.utterances['utt-6'])
        expected = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
        ]).astype(np.float32)

        assert np.array_equal(expected, actual)


class TestFrameOrdinalEncoder:

    def test_encode_utterance(self):
        ds = resources.create_multi_label_corpus()
        enc = label_encoding.FrameOrdinalEncoder(['music', 'speech', 'noise'],
                                                 frame_settings=units.FrameSettings(32000, 16000),
                                                 sr=16000)

        actual = enc.encode(ds.utterances['utt-6'])
        expected = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]).astype(np.int)

        assert np.array_equal(expected, actual)

    def test_encode_utterance_takes_larger_label(self):
        file = assets.File('file-idx', resources.sample_wav_file('wav_1.wav'))
        utt = assets.Utterance('utt-idx', file, start=0, end=8)
        ll = assets.LabelList(labels=[
            assets.Label('music', 0, 4.5),
            assets.Label('speech', 4.5, 8)
        ])
        utt.set_label_list(ll)

        enc = label_encoding.FrameOrdinalEncoder(['music', 'speech', 'noise'],
                                                 frame_settings=units.FrameSettings(32000, 16000),
                                                 sr=16000)

        actual = enc.encode(utt)
        expected = np.array([0, 0, 0, 0, 1, 1, 1]).astype(np.int)

        assert np.array_equal(expected, actual)

    def test_encode_utterance_takes_lower_index_first(self):
        file = assets.File('file-idx', resources.sample_wav_file('wav_1.wav'))
        utt = assets.Utterance('utt-idx', file, start=0, end=5)
        ll = assets.LabelList(labels=[
            assets.Label('music', 0, 3),
            assets.Label('speech', 3, 5)
        ])
        utt.set_label_list(ll)

        enc = label_encoding.FrameOrdinalEncoder(['speech', 'music', 'noise'],
                                                 frame_settings=units.FrameSettings(32000, 16000),
                                                 sr=16000)

        actual = enc.encode(utt)
        expected = np.array([1, 1, 0, 0]).astype(np.int)

        assert np.array_equal(expected, actual)
