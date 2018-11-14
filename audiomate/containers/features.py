import numpy as np

from audiomate.utils import stats
from . import container


class FeatureContainer(container.Container):
    """
    The FeatureContainer is a container for storing features
    extracted from audio data. Features are array-like data,
    where every feature represents the properties of a given segment of audio.

    Args:
        path (str): Path to where the HDF5 file is stored.
                    If the file doesn't exist, one is created.
        mode (str): Either 'r' for read-only, 'w' for truncate and write or
                    'a' for append. (default: 'a').

    Example:
        >>> fc = FeatureContainer('/path/to/hdf5file')
        >>> with fc:
        >>>     fc.set('utt-1', np.array([1,2,3,4]))
        >>>     data = fc.get('utt-1')
        array([1, 2, 3, 4])
    """

    @property
    def frame_size(self):
        """ The number of samples used per frame. """
        self.raise_error_if_not_open()
        return self._file.attrs['frame-size']

    @frame_size.setter
    def frame_size(self, frame_size):
        self.raise_error_if_not_open()
        self._file.attrs['frame-size'] = frame_size

    @property
    def hop_size(self):
        """ The number of samples between two frames. """
        self.raise_error_if_not_open()
        return self._file.attrs['hop-size']

    @hop_size.setter
    def hop_size(self, hop_size):
        self.raise_error_if_not_open()
        self._file.attrs['hop-size'] = hop_size

    @property
    def sampling_rate(self):
        """ The sampling-rate of the signal these frames are based on. """
        self.raise_error_if_not_open()
        return self._file.attrs['sampling-rate']

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate):
        self.raise_error_if_not_open()
        self._file.attrs['sampling-rate'] = sampling_rate

    def stats(self):
        """
        Return statistics calculated overall features in the container.

        Note:
            The feature container has to be opened in advance.

        Returns:
            DataStats: Statistics overall data points of all features.
        """
        self.raise_error_if_not_open()

        per_key_stats = self.stats_per_key()

        return stats.DataStats.concatenate(per_key_stats.values())

    def stats_per_key(self):
        """
        Return statistics calculated for each key in the container.

        Note:
            The feature container has to be opened in advance.

        Returns:
            dict: A dictionary containing a DataStats object for each key.
        """
        self.raise_error_if_not_open()

        all_stats = {}

        for key, data in self._file.items():
            data = data[()]
            all_stats[key] = stats.DataStats(float(np.mean(data)),
                                             float(np.var(data)),
                                             np.min(data),
                                             np.max(data),
                                             data.size)

        return all_stats
