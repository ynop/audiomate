import librosa
import audioread


class File(object):
    """
    The file object is used to hold any data/infos about a file contained in a corpus.

    Args:
        idx (str): A unique identifier within a corpus for the file.
        path (str): The path to the file.
    """
    __slots__ = ['idx', 'path']

    def __init__(self, idx, path):
        self.idx = idx
        self.path = path

    @property
    def sampling_rate(self):
        """
        Return the sampling rate.
        """
        with audioread.audio_open(self.path) as f:
            return f.samplerate

    @property
    def num_channels(self):
        """
        Return the number of channels.
        """
        with audioread.audio_open(self.path) as f:
            return f.channels

    @property
    def num_samples(self):
        """
        Return the total number of samples.
        """
        with audioread.audio_open(self.path) as f:
            return int(f.duration * f.samplerate)

    @property
    def duration(self):
        """
        Return the duration in seconds.
        """
        with audioread.audio_open(self.path) as f:
            return f.duration

    def read_samples(self, sr=None, offset=0, duration=None):
        """
        Return the samples from the file.
        Uses librosa for loading
        (see http://librosa.github.io/librosa/generated/librosa.core.load.html).

        Args:
            sr (int): If None uses the sampling rate given by the file, otherwise resamples to the given sampling rate.
            offset (float): The time in seconds, from where to start reading the samples (rel. to the file start).
            duration (float): The length of the samples to read in seconds.

        Returns:
            np.ndarray: A numpy array containing the samples as a floating point (numpy.float32) time series.
        """
        samples, __ = librosa.core.load(self.path, sr=sr, offset=offset, duration=duration)
        return samples
