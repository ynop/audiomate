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

    def test_encode_label_ends_at_utterance_end(self):
        track = tracks.FileTrack('file1', resources.sample_wav_file('med_len.wav'))
        utt = tracks.Utterance('utt1', track, start=3, end=14)
        ll = annotations.LabelList(labels=[
            annotations.Label('speech', 0, 4),
            annotations.Label('music', 4, 9),
            annotations.Label('speech', 9, float('inf')),
        ])
        utt.set_label_list(ll)

        enc = encoding.FrameHotEncoder(['music', 'speech', 'noise'],
                                       'default',
                                       frame_settings=units.FrameSettings(32000, 16000),
                                       sr=16000)

        actual = enc.encode_utterance(utt)
        expected = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ]).astype(np.float32)

        assert np.array_equal(expected, actual)

    def test_encode_label_ends_at_track_end(self):
        track = tracks.FileTrack('file1', resources.sample_wav_file('med_len.wav'))
        utt = tracks.Utterance('utt1', track, start=3, end=float('inf'))
        ll = annotations.LabelList(labels=[
            annotations.Label('speech', 0, 4),
            annotations.Label('music', 4, 9),
            annotations.Label('speech', 9, float('inf')),
        ])
        utt.set_label_list(ll)

        enc = encoding.FrameHotEncoder(['music', 'speech', 'noise'],
                                       'default',
                                       frame_settings=units.FrameSettings(32000, 16000),
                                       sr=16000)

        actual = enc.encode_utterance(utt)
        expected = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ]).astype(np.float32)

        for r in actual:
            print(r)

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
