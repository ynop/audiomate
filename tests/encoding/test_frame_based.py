import numpy as np

from audiomate import tracks
from audiomate import annotations
from audiomate.utils import units
from audiomate import encoding

from tests import resources


class TestFrameHotEncoder:

    def test_encode_full_utterance(self):
        ds = resources.create_multi_label_corpus()
        enc = encoding.FrameHotEncoder(['music', 'speech', 'noise'],
                                       'default',
                                       frame_settings=units.FrameSettings(32000, 16000),
                                       sr=16000)

        actual = enc.encode_utterance(ds.utterances['utt-6'])
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
        enc = encoding.FrameOrdinalEncoder(['music', 'speech', 'noise'],
                                           'default',
                                           frame_settings=units.FrameSettings(32000, 16000),
                                           sr=16000)

        actual = enc.encode_utterance(ds.utterances['utt-6'])
        expected = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]).astype(np.int)

        assert np.array_equal(expected, actual)

    def test_encode_utterance_takes_larger_label(self):
        file = tracks.FileTrack('file-idx', resources.sample_wav_file('wav_1.wav'))
        utt = tracks.Utterance('utt-idx', file, start=0, end=8)
        ll = annotations.LabelList(labels=[
            annotations.Label('music', 0, 4.5),
            annotations.Label('speech', 4.5, 8)
        ])
        utt.set_label_list(ll)

        enc = encoding.FrameOrdinalEncoder(['music', 'speech', 'noise'],
                                           'default',
                                           frame_settings=units.FrameSettings(32000, 16000),
                                           sr=16000)

        actual = enc.encode_utterance(utt)
        expected = np.array([0, 0, 0, 0, 1, 1, 1]).astype(np.int)

        assert np.array_equal(expected, actual)

    def test_encode_utterance_takes_lower_index_first(self):
        file = tracks.FileTrack('file-idx', resources.sample_wav_file('wav_1.wav'))
        utt = tracks.Utterance('utt-idx', file, start=0, end=5)
        ll = annotations.LabelList(labels=[
            annotations.Label('music', 0, 3),
            annotations.Label('speech', 3, 5)
        ])
        utt.set_label_list(ll)

        enc = encoding.FrameOrdinalEncoder(['speech', 'music', 'noise'],
                                           'default',
                                           frame_settings=units.FrameSettings(32000, 16000),
                                           sr=16000)

        actual = enc.encode_utterance(utt)
        expected = np.array([1, 1, 0, 0]).astype(np.int)

        assert np.array_equal(expected, actual)
