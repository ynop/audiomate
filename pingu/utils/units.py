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

    return sampling_rate / sample
