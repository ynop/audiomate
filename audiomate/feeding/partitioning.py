"""
This module module provides functionality to load data from a container into memory in chunks.
"""

import gc
import random

import numpy as np

import audiomate
from audiomate import containers
from audiomate.utils import units


class PartitioningContainerLoader(object):
    """
    Load data from one or more containers in partitions.
    It computes a scheme to load the data of as many utterances as possible in one partition.

    A scheme is initially computed on creation of the loader. To compute a new one the ``reload()`` method can be used.
    This only has an effect if ``shuffle == True``,
    otherwise the utterances are defined always loaded in the same order.

    With a given scheme, data of a partition can be retrieved via ``load_partition_data()``.
    It loads all data of the partition with the given index into memory.

    Args:
        corpus_or_utt_ids (Corpus, list): Either a corpus or a list of utterances.
                                          This defines which utterances are considered for loading.
        containers (container.Container, list): Either a single or a list of Container objects.
                                             From the given containers data is loaded.
        partition_size (str): Size of the partitions in bytes. The units ``k`` (kibibytes), ``m`` (mebibytes) and ``g``
                             (gibibytes) are supported, i.e. a ``partition_size`` of ``1g`` equates :math:`2^{30}`
                             bytes.
        shuffle (bool): Indicates whether the utterances should be returned in
                        random order (``True``) or not (``False``).
        seed (int): Seed to be used for the random number generator.

    Example:
        >>> corpus = audiomate.Corpus.load('/path/to/corpus')
        >>> container_inputs = containers.FeatureContainer('/path/to/features.hdf5')
        >>> container_outputs = containers.Container('/path/to/targets.hdf5')
        >>>
        >>> lo = PartitioningContainerLoader(corpus, [container_inputs, container_outputs], '1G', shuffle=True, seed=23)
        >>> len(lo.partitions) # Number of parititions
        5
        >>> lo.partitions[0].utt_ids # Utterances in the partition with index 0
        ['utt-1', 'utt-2', ...]
        >>> p0 = lo.load_partition_data(0) # Load partition 0 into memory
        >>> p0.info.utt_ids[0] # First utterance in the partition
        'utt-1'
        >>> p0.utt_data[0] # Data of the first utterance
        (
            array([[0.58843831, 0.18128443, 0.19718328, 0.25284105], ...]),
            array([[0.0, 1.0], ...])
        )
    """

    def __init__(self, corpus_or_utt_ids, feature_containers, partition_size,
                 shuffle=True, seed=None):
        if isinstance(corpus_or_utt_ids, audiomate.Corpus):
            self.utt_ids = list(corpus_or_utt_ids.utterances.keys())
        else:
            self.utt_ids = corpus_or_utt_ids

        if isinstance(feature_containers, containers.Container):
            self.containers = [feature_containers]
        else:
            self.containers = feature_containers

        if len(self.containers) == 0:
            raise ValueError('At least one container has to be provided!')

        self.partitions = []
        self.partition_size = units.parse_storage_size(partition_size)
        self.shuffle = shuffle

        # init random state
        self.rand = random.Random()
        self.rand.seed(a=seed)

        # check
        self._raise_error_if_container_is_missing_an_utterance()

        # Compute utterance size and length
        self.utt_sizes = self._scan()
        self.utt_lengths = self._get_all_lengths()

        self.reload()

    def reload(self):
        """
        Create a new partition scheme. A scheme defines which utterances are in which partition.
        The scheme only changes after every call if ``self.shuffle == True``.

        Returns:
            list: List of PartitionInfo objects, defining the new partitions (same as ``self.partitions``)
        """

        # Create the order in which utterances will be loaded
        utt_ids = sorted(self.utt_ids)

        if self.shuffle:
            self.rand.shuffle(utt_ids)

        partitions = []

        current_partition = PartitionInfo()

        for utt_id in utt_ids:
            utt_size = self.utt_sizes[utt_id]
            utt_lengths = self.utt_lengths[utt_id]

            # We add utterance to the partition as long the partition-size is not exceeded
            # Otherwise we start with new partition.
            if current_partition.size + utt_size > self.partition_size:
                partitions.append(current_partition)
                current_partition = PartitionInfo()

            current_partition.utt_ids.append(utt_id)
            current_partition.utt_lengths.append(utt_lengths)
            current_partition.size += utt_size

        if current_partition.size > 0:
            partitions.append(current_partition)

        self.partitions = partitions
        return self.partitions

    def load_partition_data(self, index):
        """
        Load and return the partition with the given index.

        Args:
            index (int): The index of partition, that refers to the index in ``self.partitions``.

        Returns:
            PartitionData: A PartitionData object containing the data for the partition with the given index.
        """

        info = self.partitions[index]
        data = PartitionData(info)

        for utt_id in info.utt_ids:
            utt_data = [c._file[utt_id][:] for c in self.containers]
            data.utt_data.append(utt_data)

        return data

    def _raise_error_if_container_is_missing_an_utterance(self):
        """ Check if there is a dataset for every utterance in every container, otherwise raise an error. """
        expected_keys = frozenset(self.utt_ids)

        for cnt in self.containers:
            keys = set(cnt.keys())

            if not keys.issuperset(expected_keys):
                raise ValueError('Container is missing data for some utterances!')

    def _scan(self):
        """ For every utterance, calculate the size it will need in memory. """
        utt_sizes = {}

        for dset_name in self.utt_ids:
            per_container = []

            for cnt in self.containers:
                dset = cnt._file[dset_name]
                dtype_size = dset.dtype.itemsize

                record_size = dtype_size * dset.size
                per_container.append(record_size)

            utt_size = sum(per_container)

            if utt_size > self.partition_size:
                raise ValueError('Records in "{0}" are larger than the partition size'.format(dset_name))

            utt_sizes[dset_name] = utt_size

        return utt_sizes

    def _get_all_lengths(self):
        """ For every utterance, get the length of the data in every container. Return a list of tuples. """
        utt_lengths = {}

        for utt_idx in self.utt_ids:
            per_container = [c._file[utt_idx].shape[0] for c in self.containers]
            utt_lengths[utt_idx] = tuple(per_container)

        return utt_lengths


class PartitionInfo(object):
    """
    Class for holding the info of a partition.

    Attributes:
        utt_ids (list): A list of utterance-ids in the partition.
        utt_lengths (list): List with lengths of the utterances. (Outermost dimension in the dataset of the container)
                            Since there are maybe multiple containers, every item is a tuple of lengths.
                            They correspond to the length of the utterance in every container,
                            in the order of the containers passed to the ParitioningContainerLoader.
        size (int): The number of bytes the partition will allocate, when loaded.
    """

    def __init__(self):
        self.utt_ids = []
        self.utt_lengths = []
        self.size = 0

    def total_lengths(self):
        """ Return the total length of all utterances for every container. """
        return tuple([sum(x) for x in zip(*self.utt_lengths)])


class PartitionData(object):
    """
    Class for holding the loaded data of a partition.

    Args:
        info (PartitionInfo): The info about the partition.

    Attributes:
        utt_data (list): A list holding the data-objects for every utterance in the order of ``info.utt_ids``.
                         The entries are also lists or tuples containing the array for every container.
    """

    def __init__(self, info):
        self.info = info
        self.utt_data = []


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
        >>> from audiomate.feeding import PartitioningFeatureIterator
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

    def __init__(self, hdf5file, partition_size, shuffle=True, seed=None, includes=None, excludes=None):
        self._file = hdf5file
        self._partition_size = units.parse_storage_size(partition_size)
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
