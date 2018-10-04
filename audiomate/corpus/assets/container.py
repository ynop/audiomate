import h5py


class Container(object):
    """
    A container is a wrapper around a HDF5 file.
    A container is meant to store array-like data for utterances.
    Every utterance is represented as a dataset within the HDF5 file.

    Args:
        path (str): Path where the HDF5 file is stored. If the file doesn't exist, one is created.

    Example:
        >>> ct = Container('/path/to/hdf5file')
        >>> with ct:
        >>>     ct.set('utt-1', np.array([1,2,3,4]))
        >>>     data = ct.get('utt-1')
        array([1, 2, 3, 4])
    """

    def __init__(self, path):
        self.path = path
        self._file = None

    def open(self):
        """
        Open the container file in order to read/write to it.
        """
        if self._file is None:
            self._file = h5py.File(self.path, 'a')

    def close(self):
        """
        Close the container file if its open.
        """
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def keys(self):
        """
        Return all keys available in the container.

        Returns:
            list: List of identifiers available in the container.

        Note:
            The container has to be opened in advance.
        """
        self.raise_error_if_not_open()

        return list(self._file.keys())

    def get(self, utterance_idx, mem_map=True):
        """
        Read and return the data stored for the given utterance-id.

        Args:
            utterance_idx (str): The ID of the utterance.
            mem_map (bool): If True returns the data as memory-mapped array, otherwise a copy is returned.

        Note:
            The container has to be opened in advance.

        Returns:
            numpy.ndarray: The stored data.
        """
        self.raise_error_if_not_open()

        if utterance_idx in self._file:
            data = self._file[utterance_idx]

            if not mem_map:
                data = data[()]

            return data
        else:
            return None

    def set(self, utterance_idx, data):
        """
        Set the given data to the container for the utterance with the given id.
        Any existing data of the utterance in this container is discarded/overwritten.

        Args:
            utterance_idx (str): The ID of the utterance to store the data for.
            data (numpy.ndarray): Array-like data.

        Note:
            The container has to be opened in advance.
        """
        self.raise_error_if_not_open()

        if utterance_idx in self._file:
            del self._file[utterance_idx]

        self._file.create_dataset(utterance_idx, data=data)

    def append(self, utterance_idx, data):
        """
        Append the given data to the data that already exists in the container for the given utterance.
        Only data with equal dimensions (except the first) are allowed, since they are concatenated/stacked,
        along the first dimension.

        Args:
            utterance_idx (str): The id of the utterance.
            data (numpy.ndarray): Array-like data.
                                  Has to have the same dimension as the existing data after the first dimension.

        Note:
            The container has to be opened in advance.
            For appending to existing data the HDF5-Dataset has to be chunked,
            so it is not allowed to first add data via ``set``.
        """
        existing = self.get(utterance_idx, mem_map=True)

        if existing is not None:
            num_existing = existing.shape[0]

            if existing.shape[1:] != data.shape[1:]:
                raise ValueError(
                    'The data to append needs to have the same dimensions ({}).'.format(existing.shape[1:]))

            existing.resize(num_existing + data.shape[0], 0)
            existing[num_existing:] = data
        else:
            max_shape = list(data.shape)
            max_shape[0] = None

            self._file.create_dataset(utterance_idx, data=data, chunks=True, maxshape=max_shape)

    def remove(self, utterance_idx):
        """
        Remove the data stored for the given utterance-id.

        Args:
            utterance_idx (str): ID of the utterance.

        Note:
            The container has to be opened in advance.
        """
        self.raise_error_if_not_open()

        if utterance_idx in self._file:
            del self._file[utterance_idx]

    def raise_error_if_not_open(self):
        """ Check if container is opened, raise error if not. """
        if self._file is None:
            raise ValueError('The container is not opened!')
