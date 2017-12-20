import os
import unittest

import pytest
import numpy as np
import librosa

from pingu.corpus import assets

MONO_16K_16BIT_9 = os.path.join(os.path.dirname(__file__), 'mono_16k_16bit_9.wav')
STEREO_22050_32BIT_13 = os.path.join(os.path.dirname(__file__), 'stereo_22050_32bit_13.wav')


class LabelMapperTest(unittest.TestCase):
    def setUp(self):
        self.mono_file = assets.File('fileid', MONO_16K_16BIT_9)
        self.stereo_file = assets.File('fileid', STEREO_22050_32BIT_13)

    def test_wave_info(self):
        assert self.mono_file.sampling_rate == 16000
        assert self.stereo_file.sampling_rate == 22050

    def test_num_channels(self):
        assert self.mono_file.num_channels == 1
        assert self.stereo_file.num_channels == 2

    def test_bytes_per_sample(self):
        assert self.mono_file.bytes_per_sample == 2
        assert self.stereo_file.bytes_per_sample == 4

    def test_num_samples(self):
        assert self.mono_file.num_samples == 9
        assert self.stereo_file.num_samples == 13

    def test_duration(self):
        assert self.mono_file.duration == pytest.approx(0.0005625)
        assert self.stereo_file.duration == pytest.approx(0.000589569161)

    def test_read_samples(self):
        expected, __ = librosa.core.load(MONO_16K_16BIT_9, sr=None)
        actual = self.mono_file.read_samples()
        assert np.array_equal(actual, expected)

        expected, __ = librosa.core.load(STEREO_22050_32BIT_13, sr=None)
        actual = self.stereo_file.read_samples()
        assert np.array_equal(actual, expected)
