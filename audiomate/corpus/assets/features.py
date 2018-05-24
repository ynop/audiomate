import gc
import re

import h5py
import numpy as np

from audiomate.utils import stats


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

        self._file.create_dataset(utterance_idx, data=features)

    def append(self, utterance_idx, features):
        """
        Append the given features to possible existing features of the given utterance.

        Args:
            utterance_idx (str): The id of the utterance.
            features (numpy.ndarray): A np.ndarray with the features.
                                      They have to be the same dimension as the existing ones.

        Note:
            The feature container has to be opened in advance.
            For updating the h5py-Dataset has to be chunked, so it is not allowed to first add features via ``set``.
        """
        existing = self.get(utterance_idx, mem_map=True)

        if existing is not None:
            num_existing = existing.shape[0]

            if existing.shape[1:] != features.shape[1:]:
                raise ValueError(
                    'The features to append need to have the same dimensions ({}).'.format(existing.shape[1:]))

            existing.resize(num_existing + features.shape[0], 0)
            existing[num_existing:] = features
        else:
            max_shape = list(features.shape)
            max_shape[0] = None

            self._file.create_dataset(utterance_idx, data=features, chunks=True, maxshape=max_shape)

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


class PartitioningFeatureIterator(object):
    """
    Iterates over all features in the given HDF5 file.

    Before iterating over the features, the iterator slices the file into one or more partitions and loads the data into
    memory. This leads to significant speed-ups even with moderate partition sizes, regardless of the type of disk
    (spinning or flash). Pseudo random access is supported with a negligible impact on performance and randomness: The
    data is randomly sampled (without replacement) within each partition and the partitions are loaded in random order,
    too.

    The features are emitted as triplets in the form of
    ``(utterance name, index of the feature within the utterance, feature)``.

    When calculating the partition sizes only the size of the features itself is factored in, overhead of data storage
    is ignored. This overhead is usually negligible even with partition sizes of multiple gigabytes because the data is
    stored as numpy ndarrays in memory (one per utterance). The overhead of a single ndarray is 96 bytes regardless of
    its size. Nonetheless the partition size should be chosen to be lower than the total available memory.

    Args:
        hdf5file(h5py.File): HDF5 file containing the features
        partition_size(str): Size of the partitions in bytes. The units ``k`` (kibibytes), ``m`` (mebibytes) and ``g``
                             (gibibytes) are supported, i.e. a ``partition_size`` of ``1g`` equates :math:`2^{30}`
                             bytes.
        shuffle(bool): Indicates whether the features should be returned in random order (``True``) or not (``False``).
        seed(int): Seed to be used for the random number generator.
        includes(iterable): Iterable of names of data sets that should be included when iterating over the feature
                            container. Mutually exclusive with ``excludes``. If both are specified, only ``includes``
                            will be considered.
        excludes(iterable): Iterable of names of data sets to skip when iterating over the feature container. Mutually
                            exclusive with ``includes``. If both are specified, only ``includes`` will be considered.

    Example:
        >>> import h5py
        >>> from audiomate.corpus.assets import PartitioningFeatureIterator
        >>> hdf5 = h5py.File('features.h5', 'r')
        >>> iterator = PartitioningFeatureIterator(hdf5, '12g', shuffle=True)
        >>> next(iterator)
        ('music-fma-0100', 227, array([-0.15004082, -0.30246958, -0.38708138, ..., -0.93471956,
               -0.94194776, -0.90878332], dtype=float32))
        >>> next(iterator)
        ('music-fma-0081', 2196, array([-0.00207647, -0.00101351, -0.00058832, ..., -0.00207647,
               -0.00292684, -0.00292684], dtype=float32))
        >>> next(iterator)
        ('music-hd-0050', 1026, array([-0.57352495, -0.63049972, -0.63049972, ...,  0.82490814,
                0.84680521,  0.75517786], dtype=float32))
    """

    PARTITION_SIZE_PATTERN = re.compile('^([0-9]+(\.[0-9]+)?)([gmk])?$', re.I)

    def __init__(self, hdf5file, partition_size, shuffle=True, seed=None, includes=None, excludes=None):
        self._file = hdf5file
        self._partition_size = self._parse_partition_size(partition_size)
        self._shuffle = shuffle
        self._seed = seed

        data_sets = self._filter_data_sets(hdf5file.keys(), includes=includes, excludes=excludes)
        if shuffle:
            _random_state(self._seed).shuffle(data_sets)

        self._data_sets = tuple(data_sets)
        self._partitions = []
        self._partition_idx = 0
        self._partition_data = None

        self._partition()

    def __iter__(self):
        return self

    def __next__(self):
        if self._partition_data is None or not self._partition_data.has_next():
            if self._partition_data is not None:
                self._partition_data = None
                gc.collect()  # signal gc that it's time to get rid of the obsolete data

            self._partition_data = self._load_next_partition()

            if self._partition_data is None:
                raise StopIteration

        return next(self._partition_data)

    def _load_next_partition(self):
        if len(self._partitions) == self._partition_idx:
            return None

        start, end = self._partitions[self._partition_idx]
        self._partition_idx += 1

        start_dset_name, start_idx = start
        end_dset_name, end_idx = end

        start_dset_idx = self._data_sets.index(start_dset_name)
        end_dset_idx = self._data_sets.index(end_dset_name)

        if start_dset_name == end_dset_name:
            slices = [DataSetSlice(start_dset_name, start_idx, self._file[start_dset_name][start_idx:end_idx])]
            return Partition(slices, shuffle=self._shuffle, seed=self._seed)

        slices = [DataSetSlice(start_dset_name, start_idx, self._file[start_dset_name][start_idx:])]

        middle_dsets = self._data_sets[start_dset_idx + 1:end_dset_idx]
        for dset in middle_dsets:
            slices.append(DataSetSlice(dset, 0, self._file[dset][:]))

        slices.append(DataSetSlice(end_dset_name, 0, self._file[end_dset_name][:end_idx]))

        return Partition(slices, shuffle=self._shuffle, seed=self._seed)

    def _partition(self):
        dset_props = self._scan()

        start = None
        partition_free_space = self._partition_size

        for idx, props in enumerate(dset_props):
            dset_name = props.name
            num_records = props.num_of_records
            record_size = props.record_size
            remaining_records = props.num_of_records
            is_last = (idx == len(dset_props) - 1)

            next_record_size = None if is_last else dset_props[idx + 1].record_size

            if start is None:
                start = (dset_name, 0)

            while partition_free_space >= record_size and remaining_records >= 1:
                num_fitting_records = int(partition_free_space / record_size)
                num_records_taken = min(remaining_records, num_fitting_records)
                end_index = num_records_taken if dset_name != start[0] else start[1] + num_records_taken
                end = (dset_name, end_index)

                if num_records_taken == num_fitting_records:  # Partition is going to be full afterwards
                    self._partitions.append((start, end))

                    partition_free_space = self._partition_size

                    if end[1] == num_records:  # Data set is exhausted
                        start = None
                        break
                    else:  # Next partition starts within the same data set
                        start = end
                elif num_records_taken == remaining_records and is_last:  # All data sets are partitioned
                    self._partitions.append((start, end))
                    break
                else:
                    partition_free_space -= record_size * num_records_taken

                    if partition_free_space < next_record_size:
                        self._partitions.append((start, end))
                        start = None
                        partition_free_space = self._partition_size
                        break

                remaining_records -= num_records_taken

        if self._shuffle:
            _random_state(self._seed).shuffle(self._partitions)

    def _scan(self):
        dset_props = []

        for dset_name in self._data_sets:
            dtype_size = self._file[dset_name].dtype.itemsize

            if len(self._file[dset_name]) == 0:
                continue

            num_records, items_per_record = self._file[dset_name].shape
            record_size = dtype_size * items_per_record

            if record_size > self._partition_size:
                raise ValueError('Records in "{0}" are larger than the partition size'.format(dset_name))

            dset_props.append(DataSetProperties(dset_name, num_records, record_size))

        return dset_props

    @staticmethod
    def _parse_partition_size(partition_size):
        units = {
            'k': 1024,
            'm': 1024 * 1024,
            'g': 1024 * 1024 * 1024
        }

        match = PartitioningFeatureIterator.PARTITION_SIZE_PATTERN.fullmatch(str(partition_size))

        if match is None:
            raise ValueError('Invalid partition size: {0}'.format(partition_size))

        groups = match.groups()

        if groups[2] is None:  # no units
            return int(float(groups[0]))  # silently dropping the float, because byte is the smallest unit)

        return int(float(groups[0]) * units[groups[2].lower()])

    @staticmethod
    def _filter_data_sets(data_sets, includes=None, excludes=None):
        if includes is None:
            includes = frozenset()
        else:
            includes = frozenset(includes)

        if excludes is None:
            excludes = frozenset()
        else:
            excludes = frozenset(excludes)

        if len(includes) > 0:
            return [data_set for data_set in data_sets if data_set in includes]

        return [data_set for data_set in data_sets if data_set not in excludes]


class DataSetProperties:
    def __init__(self, name, num_of_records, record_size):
        self.name = name
        self.num_of_records = num_of_records
        self.record_size = record_size

    def __repr__(self):
        return 'DataSetProperties({0}, {1}, {2})'.format(self.name, self.num_of_records, self.record_size)


class Partition:
    def __init__(self, slices, shuffle=True, seed=None):
        self._slices = slices

        self._total_length = 0
        for item in slices:
            self._total_length += item.length

        self._index = 0

        if shuffle:
            self._elements = _random_state(seed).permutation(self._total_length)
        else:
            self._elements = np.arange(0, self._total_length)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == self._total_length:
            raise StopIteration()

        index = self._elements[self._index]
        for item in self._slices:
            if index >= item.length:
                index -= item.length
                continue

            self._index += 1

            # emits triplet (data set's name, original index of feature within data set, feature)
            return item.data_set_name, item.start_index + index, item.data[index]

    def has_next(self):
        return self._index < self._total_length


class DataSetSlice:
    def __init__(self, data_set_name, start_index, data):
        self.data_set_name = data_set_name
        self.start_index = start_index
        self.length = len(data)
        self.data = data


def _random_state(seed=None):
    random_state = np.random.RandomState()

    if seed is not None:
        random_state.seed(seed)

    return random_state
