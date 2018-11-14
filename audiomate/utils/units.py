"""
This module contains functions for handling different units.
Especially it provides function to convert from one to another unit (e.g. seconds -> sample-indexn).
"""

import math
import re

import numpy as np


def parse_storage_size(storage_size):
    """
    Parses an expression that represents an amount of storage/memory and returns the number of bytes it represents.

    Args:
        storage_size(str): Size in bytes. The units ``k`` (kibibytes), ``m`` (mebibytes) and ``g``
                           (gibibytes) are supported, i.e. a ``partition_size`` of ``1g`` equates :math:`2^{30}` bytes.

    Returns:
        int: Number of bytes.
    """
    pattern = re.compile(r'^([0-9]+(\.[0-9]+)?)([gmk])?$', re.I)

    units = {
        'k': 1024,
        'm': 1024 * 1024,
        'g': 1024 * 1024 * 1024
    }

    match = pattern.fullmatch(str(storage_size))

    if match is None:
        raise ValueError('Invalid partition size: {0}'.format(storage_size))

    groups = match.groups()

    # no units
    if groups[2] is None:
        # silently dropping the float, because byte is the smallest unit)
        return int(float(groups[0]))

    return int(float(groups[0]) * units[groups[2].lower()])


def seconds_to_sample(seconds, sampling_rate=16000):
    """
    Convert a value in seconds to a sample index based on the given sampling rate.

    Args:
        seconds (float): The value in seconds.
        sampling_rate (int): The sampling rate to use for conversion.

    Returns:
        int: The sample index (0-based).

    Example::
        >>> seconds_to_sample(1.3, sampling_rate=16000)
        20800
    """

    return int(np.round(sampling_rate * seconds))


def sample_to_seconds(sample, sampling_rate=16000):
    """
    Convert a sample index to seconds based on the given sampling rate.

    Args:
        sample (int): The index of the sample (0 based).
        sampling_rate (int): The sampling rate to use for conversion.

    Returns:
        float: The time in seconds.

    Example::
        >>> sample_to_seconds(20800, sampling_rate=16000)
        1.3
    """

    return sample / sampling_rate


class FrameSettings(object):
    """
    This class provides functions for handling conversions/calculations between time, samples and frames.

    By default the framing is done as follows:
        * The first frame starts at sample 0
        * The end of the last frame is higher than the last sample.
        * The end of the last frame is smaller than the last sample + hop_size

    Args:
        frame_size (int): Number of samples used per frame.
        hop_size (int): Number of samples between two frames.
    """

    def __init__(self, frame_size, hop_size):
        self.frame_size = frame_size
        self.hop_size = hop_size

    def num_frames(self, num_samples):
        """
        Return the number of frames that will be used for a signal with the length of ``num_samples``.
        """
        return math.ceil(float(max(num_samples - self.frame_size, 0)) / float(self.hop_size)) + 1

    def sample_to_frame_range(self, sample_index):
        """
        Return a tuple containing the indices of the first frame containing the sample with the given index and
        the last frame (exclusive, doesn't contain the sample anymore).
        """
        start = max(0, int((sample_index - self.frame_size) / self.hop_size) + 1)
        end = int(sample_index / self.hop_size) + 1
        return start, end

    def frame_to_sample(self, frame_index):
        """
        Return a tuple containing the indices of the sample which are the first sample and the end (exclusive)
        of the frame with the given index.
        """
        start = frame_index * self.hop_size
        end = start + self.frame_size
        return start, end

    def frame_to_seconds(self, frame_index, sr):
        """
        Return a tuple containing the start and end of the frame in seconds.
        """
        start_sample, end_sample = self.frame_to_sample(frame_index)
        return sample_to_seconds(start_sample, sampling_rate=sr), sample_to_seconds(end_sample, sampling_rate=sr)

    def time_range_to_frame_range(self, start, end, sr):
        """
        Calculate the frames containing samples from the given time range in seconds.

        Args:
            start (float): Start time in seconds.
            end (float): End time in seconds.
            sr (int): The sampling rate to use for time-to-sample conversion.

        Returns:
            tuple: A tuple containing the start and end (exclusive) frame indices.
        """

        start_sample = seconds_to_sample(start, sr)
        end_sample = seconds_to_sample(end, sr)

        return self.sample_to_frame_range(start_sample)[0], self.sample_to_frame_range(end_sample - 1)[1]
