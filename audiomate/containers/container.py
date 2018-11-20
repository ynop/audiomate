import contextlib
import h5py


class Container(object):
    """
    A container is a wrapper around a HDF5 file.
    In a container is used to store array-like data.
    Every array is associated with some idx/key.
    Every array (a dataset in h5py-terms) may have additional attributes.

    Args:
        path (str): Path where the HDF5 file is stored.
                    If the file doesn't exist, one is created.
        mode (str): Either 'r' for read-only, 'w' for truncate and write or
                    'a' for append. (default: 'a').

    Example:
        >>> ct = Container('/path/to/hdf5file')
        >>> with ct:
        >>>     ct.set('utt-1', np.array([1,2,3,4]))
        >>>     data = ct.get('utt-1')
        array([1, 2, 3, 4])
    """

    def __init__(self, path, mode='a'):
        if mode not in ['r', 'w', 'a']:
            raise ValueError('Invalid mode! Modes: [\'a\', \'r\', \'w\']')

        self.path = path
        self.mode = mode

        self._file = None

    def open(self, mode=None):
        """
        Open the container file.

        Args:
            mode (str): Either 'r' for read-only, 'w' for truncate and write or
                        'a' for append. (default: 'a').
                        If ``None``, uses ``self.mode``.
        """

        if mode is None:
            mode = self.mode
        elif mode not in ['r', 'w', 'a']:
            raise ValueError('Invalid mode! Modes: [\'a\', \'r\', \'w\']')

        if self._file is None:
            self._file = h5py.File(self.path, mode=mode)

    def close(self):
        """
        Close the container file if its open.
        """
        if self._file is not None:
            self._file.close()
            self._file = None

    def is_open(self, mode=None):
        """
        Return ``True``, if container is already open. ``False`` otherwise.
        """
        return self._file is not None

    @contextlib.contextmanager
    def open_if_needed(self, mode=None):
        """
        Convenience context-manager for the use with ``with``.
        Opens the container if not already done.
        Only closes the container if it was opened within this context.

        Args:
            mode (str): Either 'r' for read-only, 'w' for truncate and write or
                        'a' for append. (default: 'a').
                        If ``None``, uses ``self.mode``.
        """
        was_open = self.is_open()

        if not was_open:
            self.open(mode=mode)

        try:
            yield self
        finally:
            if not was_open:
                self.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def keys(self):
        """
        Return a list of keys for which an array is stored in the container.

        Returns:
            list: List of identifiers available in the container.

        Note:
            The container has to be opened in advance.
        """
        self.raise_error_if_not_open()

        return sorted(list(self._file.keys()))

    def get(self, key, mem_map=True):
        """
        Read and return the data stored for the given key.

        Args:
            key (str): The key to read the data from.
            mem_map (bool): If ``True`` returns the data as
                            memory-mapped array, otherwise a copy is returned.

        Note:
            The container has to be opened in advance.

        Returns:
            numpy.ndarray: The stored data.
        """
        self.raise_error_if_not_open()

        if key in self._file:
            data = self._file[key]

            if not mem_map:
                data = data[()]

            return data
        else:
            return None

    def set(self, key, data):
        """
        Set the given data to the container with the given key.
        Any existing data for the given key is discarded/overwritten.

        Args:
            key (str): A key to store the data for.
            data (numpy.ndarray): Array-like data.

        Note:
            The container has to be opened in advance.
        """
        self.raise_error_if_not_open()

        if key in self._file:
            del self._file[key]

        self._file.create_dataset(key, data=data)

    def append(self, key, data):
        """
        Append the given data to the data that already exists
        in the container for the given key.
        Only data with equal dimensions (except the first) are allowed,
        since they are concatenated/stacked along the first dimension.

        Args:
            key (str): Key to store data for.
            data (numpy.ndarray): Array-like data.
                                  Has to have the same dimension as
                                  the existing data after the first dimension.

        Note:
            The container has to be opened in advance.
            For appending to existing data the HDF5-Dataset has to be chunked,
            so it is not allowed to first add data via ``set``.
        """
        existing = self.get(key, mem_map=True)

        if existing is not None:
            num_existing = existing.shape[0]

            if existing.shape[1:] != data.shape[1:]:
                error_msg = (
                    'The data to append needs to'
                    'have the same dimensions ({}).'
                )
                raise ValueError(error_msg.format(existing.shape[1:]))

            existing.resize(num_existing + data.shape[0], 0)
            existing[num_existing:] = data
        else:
            max_shape = list(data.shape)
            max_shape[0] = None

            self._file.create_dataset(key, data=data,
                                      chunks=True, maxshape=max_shape)

    def remove(self, key):
        """
        Remove the data stored for the given key.

        Args:
            key (str): Key of the data to remove.

        Note:
            The container has to be opened in advance.
        """
        self.raise_error_if_not_open()

        if key in self._file:
            del self._file[key]

    def raise_error_if_not_open(self):
        """ Check if container is opened, raise error if not. """
        if self._file is None:
            raise ValueError('The container is not opened!')
