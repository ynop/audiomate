import abc


class Track(abc.ABC):
    """
    Track is the abstract base class for an audio track.

    Args:
        idx (str): A identifier to uniquely identify a track.
    """
    __slots__ = ['idx']

    def __init__(self, idx):
        self.idx = idx

    @property
    @abc.abstractmethod
    def sampling_rate(self):
        """
        Return the sampling rate.
        """
        pass

    @property
    @abc.abstractmethod
    def num_channels(self):
        """
        Return the number of channels.
        """
        pass

    @property
    @abc.abstractmethod
    def num_samples(self):
        """
        Return the total number of samples.
        """
        pass

    @property
    @abc.abstractmethod
    def duration(self):
        """
        Return the duration in seconds.
        """
        pass

    @abc.abstractmethod
    def read_samples(self, sr=None, offset=0, duration=None):
        """
        Return the samples from the file.

        Args:
            sr (int): If ``None``, uses the native sampling-rate,
                      otherwise resamples to the given sampling rate.
            offset (float): The time in seconds, from where to start
                            reading the samples (rel. to the track start).
            duration (float): The length of the samples to read in seconds.

        Returns:
            np.ndarray: A numpy array containing the samples
                        as a floating point (numpy.float32) time series.
        """
        pass
