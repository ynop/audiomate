import pytest

from audiomate.utils import units


def test_partition_size_in_bytes_specified_as_int():
    assert 1024 == units.parse_storage_size(1024)


def test_partition_size_in_bytes():
    assert 1024 == units.parse_storage_size('1024')


def test_partition_size_in_kibibytes():
    assert 2 * 1024 == units.parse_storage_size('2k')


def test_partition_size_in_mebibytes():
    assert 2 * 1024 * 1024 == units.parse_storage_size('2m')


def test_partition_size_in_gibibytes():
    assert 2 * 1024 * 1024 * 1024 == units.parse_storage_size('2g')


def test_partition_size_in_gibibytes_with_capital_g():
    assert 2 * 1024 * 1024 * 1024 == units.parse_storage_size('2G')


def test_partition_size_fractions_of_bytes_are_ignored():
    assert 1 == units.parse_storage_size('1.1')


def test_partition_size_half_a_gibibyte():
    assert 512 * 1024 * 1024 == units.parse_storage_size('0.5g')


class TestUnits:
    def test_seconds_to_sample(self):
        assert units.seconds_to_sample(0, 16000) == 0
        assert units.seconds_to_sample(4.5, 22050) == 99225

    def test_sample_to_seconds(self):
        assert units.sample_to_seconds(0, 44100) == 0
        assert units.sample_to_seconds(3000, 8000) == 0.375


class TestFrameSettings:

    @pytest.mark.parametrize('frame_size,hop_size,num_samples,num_frames', [
        (4, 2, 12, 5),
        (4, 2, 13, 6),
        (4, 2, 15, 7),
        (4, 2, 16, 7),
        (4096, 2048, 41523, 20),
        (32000, 16000, 240000, 14)
    ])
    def test_num_frames(self, frame_size, hop_size, num_samples, num_frames):
        f = units.FrameSettings(frame_size, hop_size)
        assert f.num_frames(num_samples) == num_frames

    @pytest.mark.parametrize('frame_size,hop_size,frame_index,start_sample,end_sample', [
        (400, 160, 0, 0, 400),
        (400, 160, 5, 800, 1200),
    ])
    def test_frame_to_sample(self, frame_size, hop_size, frame_index, start_sample, end_sample):
        f = units.FrameSettings(frame_size, hop_size)
        assert f.frame_to_sample(frame_index) == (start_sample, end_sample)

    @pytest.mark.parametrize('frame_size,hop_size,sample_index,start_frame,end_frame', [
        (4, 2, 0, 0, 1),
        (4, 2, 1, 0, 1),
        (4, 2, 2, 0, 2),
        (4, 2, 5, 1, 3)
    ])
    def test_sample_to_frame_range(self, frame_size, hop_size, sample_index, start_frame, end_frame):
        f = units.FrameSettings(frame_size, hop_size)
        assert f.sample_to_frame_range(sample_index) == (start_frame, end_frame)

    @pytest.mark.parametrize('frame_size,hop_size,frame_index,sr,start,end', [
        (400, 160, 13, 16000, 0.13, 0.155)
    ])
    def test_frame_to_seconds(self, frame_size, hop_size, frame_index, sr, start, end):
        f = units.FrameSettings(frame_size, hop_size)
        assert f.frame_to_seconds(frame_index, sr) == (start, end)

    @pytest.mark.parametrize('frame_size,hop_size,start_time,end_time,sr,start_index,end_index', [
        (400, 160, 0.37, 2.84, 8000, 17, 142),
        (32000, 16000, 0.0, 5.0, 16000, 0, 5),
        (32000, 16000, 13.0, 15.0, 16000, 12, 15),
    ])
    def test_time_range_to_frame_range(self, frame_size, hop_size, start_time, end_time, sr, start_index, end_index):
        f = units.FrameSettings(frame_size, hop_size)
        assert f.time_range_to_frame_range(start_time, end_time, sr) == (start_index, end_index)
