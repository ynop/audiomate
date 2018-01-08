import unittest

from pingu.utils import units


class ConvertionTest(unittest.TestCase):
    def test_seconds_to_sample(self):
        assert units.seconds_to_sample(0, 16000) == 0
        assert units.seconds_to_sample(4.5, 22050) == 99225

    def test_sample_to_seconds(self):
        assert units.sample_to_seconds(0, 44100) == 0
        assert units.sample_to_seconds(3000, 8000) == 0.375

    def test_sample_to_frame(self):
        assert units.sample_to_frame(0, 1024) == 0
        assert units.sample_to_frame(44200, 1024) == 43

    def test_seconds_to_frame(self):
        assert units.seconds_to_frame(0, 512) == 0
        assert units.seconds_to_frame(4.9, 256) == 306
