import wave
import librosa


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
        with wave.open(self.path, 'r') as f:
            return f.getframerate()

    @property
    def num_channels(self):
        """
        Return the number of channels.
        """
        with wave.open(self.path, 'r') as f:
            return f.getnchannels()

    @property
    def bytes_per_sample(self):
        """
        Return the number of bytes per sample.
        """
        with wave.open(self.path, 'r') as f:
            return f.getsampwidth()

    @property
    def num_samples(self):
        """
        Return the total number of samples.
        """
        with wave.open(self.path, 'r') as f:
            return f.getnframes()

    @property
    def duration(self):
        """
        Return the duration in seconds.
        """
        with wave.open(self.path, 'r') as f:
            return f.getnframes() / f.getframerate()

    def read_samples(self, sr=None):
        """
        Return the samples from the file.
        Uses librosa for loading
        (see http://librosa.github.io/librosa/generated/librosa.core.load.html).

        Args:
            sr (int): If None uses the sampling rate given by the file,
            otherwise resamples to the given sampling rate.

        Returns:
            np.ndarray: A numpy array containing the samples.
        """
        samples, __ = librosa.core.load(self.path, sr=sr)
        return samples
