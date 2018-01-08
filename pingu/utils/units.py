"""
This module contains functions for handling different units.
Especially it provides function to convert from one to another unit (e.g. seconds -> sample-indexn).
"""


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

    return int(seconds * sampling_rate)


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


def sample_to_frame(sample, hop_size):
    """
    Convert a sample index to a frame index.

    Args:
        sample (int): The index of the sample (0 based).
        hop_size (int): The number of samples between two frames.

    Returns:
        int: The frame index.
    """
    return sample // hop_size


def seconds_to_frame(seconds, hop_size, sampling_rate=16000):
    """
    Convert a time in seconds to the frame index.

    Args:
        seconds (float): The time in seconds.
        hop_size (int): The number of samples between two frames.
        sampling_rate (int): The sampling rate of the signal.
    """
    sample = seconds_to_sample(seconds, sampling_rate=sampling_rate)
    return sample_to_frame(sample, hop_size)
