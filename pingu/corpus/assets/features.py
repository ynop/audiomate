import h5py
import numpy as np

from pingu.utils import stats


class FeatureContainer(object):
    """
    A feature-container holds matrix-like data. The data is stored as HDF5 file.
    The feature-container provides functionality to access this data. For each utterance a hdf5
    data set is created within the file, if there is feature-data for a given utterance.

    Args:
        path (str): Path to where the HDF5 file is stored. If the file doesn't exist, one is
                    created.

    Examples::

        >>> fc = FeatureContainer('/path/to/hdf5file')

        >>> with fc:
        >>>     fc.set('utt-1', np.array([1,2,3,4]))
        >>>     data = fc.get('utt-1')
        array([1, 2, 3, 4])
    """

    def open(self):
        """
        Open the feature container file in order to read/write to it.
        """
        if self._file is None:
            self._file = h5py.File(self.path, 'a')

    def close(self):
        """
        Close the feature container file if its open.
        """
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __init__(self, path):
        self.path = path
        self._file = None

    @property
    def frame_size(self):
        """ The number of samples used per frame. """
        self._check_is_open()
        return self._file.attrs['frame-size']

    @frame_size.setter
    def frame_size(self, frame_size):
        self._check_is_open()
        self._file.attrs['frame-size'] = frame_size

    @property
    def hop_size(self):
        """ The number of samples between two frames. """
        self._check_is_open()
        return self._file.attrs['hop-size']

    @hop_size.setter
    def hop_size(self, hop_size):
        self._check_is_open()
        self._file.attrs['hop-size'] = hop_size

    @property
    def sampling_rate(self):
        """ The sampling-rate of the signal these frames are based on. """
        self._check_is_open()
        return self._file.attrs['sampling-rate']

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate):
        self._check_is_open()
        self._file.attrs['sampling-rate'] = sampling_rate

    def keys(self):
        """
        Return all keys available in the feature-container.

        Returns:
            keys (list): List of identifiers available in the feature-container.

        Note:
            The feature container has to be opened in advance.
        """
        self._check_is_open()

        return list(self._file.keys())

    def set(self, utterance_idx, features):
        """
        Add the given feature matrix to the feature container for the utterance with the given id.
        Any existing features of the utterance in this container are discarded/overwritten.

        Args:
            utterance_idx (str): The ID of the utterance to store the features for.
            features (numpy.ndarray): A np.ndarray with the features.

        Note:
            The feature container has to be opened in advance.
        """
        self._check_is_open()

        if utterance_idx in self._file:
            del self._file[utterance_idx]

        self._file.create_dataset(utterance_idx, data=features, compression='lzf')

    def remove(self, utterance_idx):
        """
        Remove the features stored for the given utterance-id.

        Args:
            utterance_idx (str): ID of the utterance.

        Note:
            The feature container has to be opened in advance.
        """
        self._check_is_open()

        if utterance_idx in self._file:
            del self._file[utterance_idx]

    def get(self, utterance_idx, mem_map=True):
        """
        Read and return the features stored for the given utterance-id.

        Args:
            utterance_idx (str): The ID of the utterance to get the feature-matrix from.
            mem_map (bool): If True returns the features as memory-mapped array, otherwise a copy is returned.

        Note:
            The feature container has to be opened in advance.

        Returns:
            numpy.ndarray: The stored data.
        """
        self._check_is_open()

        if utterance_idx in self._file:
            data = self._file[utterance_idx]

            if not mem_map:
                data = data[()]

            return data
        else:
            return None

    def stats(self):
        """
        Return statistics calculated overall features in the container.

        Note:
            The feature container has to be opened in advance.

        Returns:
            DataStats: Statistics overall data points of all features.
        """
        self._check_is_open()

        per_utt_stats = self.stats_per_utterance()

        return stats.DataStats.concatenate(per_utt_stats.values())

    def stats_per_utterance(self):
        """
        Return statistics calculated for each utterance in the container.

        Note:
            The feature container has to be opened in advance.

        Returns:
            dict: A dictionary containing a DataStats object for each utterance.
        """
        self._check_is_open()

        all_stats = {}

        for utt_id, data in self._file.items():
            data = data[()]
            all_stats[utt_id] = stats.DataStats(float(np.mean(data)),
                                                float(np.var(data)),
                                                np.min(data),
                                                np.max(data),
                                                data.size)

        return all_stats

    def _check_is_open(self):
        if self._file is None:
            raise ValueError('The feature container is not opened!')
