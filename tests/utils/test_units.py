import unittest

from pingu.utils import units


class UnitsTest(unittest.TestCase):
    def test_seconds_to_sample(self):
        assert units.seconds_to_sample(0, 16000) == 0
        assert units.seconds_to_sample(4.5, 22050) == 99225

    def test_sample_to_seconds(self):
        assert units.sample_to_seconds(0, 44100) == 0
        assert units.sample_to_seconds(3000, 8000) == 0.375


class FrameSettingsTest(unittest.TestCase):

    def test_num_frames(self):
        f = units.FrameSettings(4, 2)
        assert f.num_frames(12) == 5
        assert f.num_frames(13) == 6
        assert f.num_frames(15) == 7

    def test_frame_to_start_sample(self):
        f = units.FrameSettings(400, 160)
        assert f.frame_to_sample(0) == (0, 400)
        assert f.frame_to_sample(5) == (800, 1200)

    def test_sample_to_frame_range(self):
        f = units.FrameSettings(4, 2)
        assert f.sample_to_frame_range(0) == (0, 1)
        assert f.sample_to_frame_range(1) == (0, 1)
        assert f.sample_to_frame_range(2) == (0, 2)
        assert f.sample_to_frame_range(5) == (1, 3)

    def test_time_range_to_frame_range(self):
        f = units.FrameSettings(400, 160)
        assert f.time_range_to_frame_range(0.37, 2.84, 8000) == (17, 142)

    def test_time_range_to_frame_range_on_frame_boundary(self):
        f = units.FrameSettings(32000, 16000)
        assert f.time_range_to_frame_range(0, 5, 16000) == (0, 5)
