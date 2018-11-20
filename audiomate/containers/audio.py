import numpy as np

from . import container

SAMPLING_RATE_ATTR = 'sampling-rate'
MAX_INT16_VALUE = np.iinfo(np.int16).max


class AudioContainer(container.Container):
    """
    Container to store raw audio samples.

    Notes:
        The samples are stored  as 16-Bit Integers.
        But all methods expect or return the samples as 32-Bit Floats,
        in the range of -1.0 to 1.0.
    """

    def get(self, key, mem_map=True):
        """
        Return the samples for the given key and the sampling-rate.

        Args:
            key (str): The key to read the data from.
            mem_map (bool): If ``True`` returns the data as
                            memory-mapped array, otherwise a copy is returned.

        Note:
            The container has to be opened in advance.

        Returns:
            tuple: A tuple containing the samples as numpy array
                   with ``np.float32`` [-1.0,1.0] and the sampling-rate.
        """
        self.raise_error_if_not_open()

        if key in self._file:
            data = self._file[key]
            sampling_rate = data.attrs[SAMPLING_RATE_ATTR]

            if not mem_map:
                data = data[()]

            data = np.float32(data) / MAX_INT16_VALUE

            return data, sampling_rate

    def set(self, key, samples, sampling_rate):
        """
        Set the samples and sampling-rate for the given key.
        Existing data will be overwritten.
        The samples have to have ``np.float32`` datatype and values in
        the range of -1.0 and 1.0.

        Args:
            key (str): A key to store the data for.
            samples (numpy.ndarray): 1-D array of audio samples (np.float32).
            sampling_rate (int): The sampling-rate of the audio samples.

        Note:
            The container has to be opened in advance.
        """
        if not np.issubdtype(samples.dtype, np.floating):
            raise ValueError('Samples are required as np.float32!')

        if len(samples.shape) > 1:
            raise ValueError('Only single channel supported!')

        self.raise_error_if_not_open()

        if key in self._file:
            del self._file[key]

        samples = (samples * MAX_INT16_VALUE).astype(np.int16)

        dset = self._file.create_dataset(key, data=samples)
        dset.attrs[SAMPLING_RATE_ATTR] = sampling_rate

    def append(self, key, samples, sampling_rate):
        """
        Append the given samples to the data that already exists
        in the container for the given key.

        Args:
            key (str): A key to store the data for.
            samples (numpy.ndarray): 1-D array of audio samples (int-16).
            sampling_rate (int): The sampling-rate of the audio samples.

        Note:
            The container has to be opened in advance.
            For appending to existing data the HDF5-Dataset has to be chunked,
            so it is not allowed to first add data via ``set``.
        """
        if not np.issubdtype(samples.dtype, np.floating):
            raise ValueError('Samples are required as np.float32!')

        if len(samples.shape) > 1:
            raise ValueError('Only single channel supported!')

        existing = self.get(key, mem_map=True)
        samples = (samples * MAX_INT16_VALUE).astype(np.int16)

        if existing is not None:
            existing_samples, existing_sr = existing

            if existing_sr != sampling_rate:
                raise ValueError('Different sampling-rate than existing data!')

            num_existing = existing_samples.shape[0]
            self._file[key].resize(num_existing + samples.shape[0], 0)
            self._file[key][num_existing:] = samples
        else:
            dset = self._file.create_dataset(key, data=samples,
                                             chunks=True, maxshape=(None,))

            dset.attrs[SAMPLING_RATE_ATTR] = sampling_rate
