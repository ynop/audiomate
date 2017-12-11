import os
import unittest

import pytest
import numpy as np

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
        ref = np.array([6.10351562e-05, 9.15527344e-05, 1.22070312e-04,
                        1.52587891e-04, 1.03759766e-03, 1.83105469e-04,
                        2.13623047e-04, 9.15527344e-05, 1.52587891e-04], dtype=np.float32)

        assert np.array_equal(ref, self.mono_file.read_samples())

        ref = np.array([7.62939453e-05, -3.05175781e-05, -6.10351562e-05,
                        3.05175781e-05, -9.15527344e-05, 3.05175781e-05,
                        1.52587891e-05, 1.83105469e-04, -1.06811523e-04,
                        -3.05175781e-05, -1.67846680e-04, -6.10351562e-05,
                        1.37329102e-04], dtype=np.float32)

        assert np.array_equal(ref, self.stereo_file.read_samples())
