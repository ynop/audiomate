import h5py


class FeatureContainer(object):
    """
    A feature-container holds matrix-like data. The data is stored as HDF5 file.
    The feature-container provides functionality to access this data. For each utterance a hdf5 dataset is created within the file, if there is feature-data for a given utterance.

    Args:
        path (str): Path to where the HDF5 file is stored. If the file doesn't exist, one is created.

    Examples::

        >>> fc = FeatureContainer('/path/to/hdf5file')

        >>> with fc:
        >>>     fc.set('utt-1', np.array([1,2,3,4]))
        >>>     data = fc.get('utt-1')
        array([1, 2, 3, 4])
    """

    def __init__(self, path):
        self.path = path
        self._file = None

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
        if self._file is None:
            raise ValueError("The feature container is not opened!")

        if utterance_idx in self._file:
            del self._file[utterance_idx]

        self.file.create_dataset(utterance_idx, data=features, compression="lzf")

    def remove(self, utterance_idx):
        """
        Remove the features stored for the given utterance-id.

        Args:
            utterance_idx (str): ID of the utterance.

        Note:
            The feature container has to be opened in advance.
        """
        if self._file is None:
            raise ValueError("The feature container is not opened!")

        if utterance_idx in self._file:
            del self._file[utterance_idx]

    def get(self, utterance_idx):
        """
        Read and return the features stored for the given utterance-id.

        Args:
            utterance_idx: The ID of the utterance to get the feature-matrix from.

        Note:
            The feature container has to be opened in advance.

        Returns:
            numpy.ndarray: The stored data.
        """
        if self._file is None:
            raise ValueError("The feature container is not opened!")

        if utterance_idx in self._file:
            return self._file[utterance_idx][()]
        else:
            return None
