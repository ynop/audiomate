import bisect
import random

import numpy as np

import audiomate
from audiomate.corpus import assets
from . import partitioning


class DataIterator(object):
    """
    An abstract class representing a data-iterator. A data-iterator provides sequential access to data.
    An implementation of a concrete data-iterator should override the methods ``__iter__`` and ``__next__``.

    A sample returned from a data-iterator is a tuple containing the data for this sample from every container.
    The data from different containers is ordered in the way the containers were passed to the DataIterator.

    Args:
        corpus_or_utt_ids (Corpus, list): Either a corpus or a list of utterances.
                                          This defines which utterances are considered for iterating.
        containers (list, Container): A single container or a list of containers.
        shuffle (bool): Indicates whether the data should be returned in
                        random order (``True``) or not (``False``).
        seed (int): Seed to be used for the random number generator.
    """

    def __init__(self, corpus_or_utt_ids, containers, shuffle=True, seed=None):
        if isinstance(corpus_or_utt_ids, audiomate.Corpus):
            self.utt_ids = list(corpus_or_utt_ids.utterances.keys())
        else:
            self.utt_ids = corpus_or_utt_ids

        if isinstance(containers, assets.Container):
            self.containers = [containers]
        else:
            self.containers = containers

        if len(self.containers) == 0:
            raise ValueError('At least one container has to be provided!')

        self.shuffle = shuffle

        self.rand = random.Random()
        self.rand.seed(a=seed)

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class FrameIterator(DataIterator):
    """
    A data-iterator wrapping frames of a corpus. A single sample represents a single frame.

    Args:
        corpus_or_utt_ids (Corpus, list): Either a corpus or a list of utterances.
                                          This defines which utterances are considered for iterating.
        container (list, Container): A single container or a list of containers.
        partition_size (str): Size of the partitions in bytes. The units ``k`` (kibibytes), ``m`` (mebibytes) and ``g``
                              (gibibytes) are supported, i.e. a ``partition_size`` of ``1g`` equates :math:`2^{30}`
                              bytes.
        shuffle (bool): Indicates whether the data should be returned in
                        random order (``True``) or not (``False``).
        seed (int): Seed to be used for the random number generator.
    Note:
        For a FrameIterator it is expected that every container contains exactly one value/vector for every frame.
        So the first dimension of every array in every container have to match.
    """

    def __init__(self, corpus_or_utt_ids, containers, partition_size, shuffle=True, seed=None):
        super(FrameIterator, self).__init__(corpus_or_utt_ids, containers, shuffle=shuffle, seed=seed)

        self.partition_size = partition_size

        self.loader = None
        self.current_partition = None
        self.current_partition_index = -1
        self.current_frame_index = 0

        self.loader = partitioning.PartitioningContainerLoader(self.utt_ids,
                                                               self.containers,
                                                               self.partition_size,
                                                               shuffle=self.shuffle,
                                                               seed=self.rand.random())

    def __iter__(self):
        self.current_partition = None
        self.current_partition_index = -1
        self.current_frame_index = 0

        self.loader.reload()

        return self

    def __next__(self):
        if self.current_partition is None or self.current_frame_index >= len(self.current_partition):
            self.current_partition_index += 1
            self.current_frame_index = 0

            if self.current_partition_index < len(self.loader.partitions):
                partition_data = self.loader.load_partition_data(self.current_partition_index)
                self.current_partition = FramePartitionData(partition_data,
                                                            shuffle=self.shuffle,
                                                            seed=self.rand.random())
            else:
                raise StopIteration

        next_frame = self.current_partition[self.current_frame_index]
        self.current_frame_index += 1

        return next_frame


class FramePartitionData(object):
    """
    Wrapper for PartitionData to access the frames via indexes.

    Args:
        partition_data (PartitionData): The loaded partition-data.
        shuffle (bool): If True the frames are shuffled randomly for access.
        seed (int): The seed to use for shuffling.
    """

    def __init__(self, partition_data, shuffle=True, seed=None):
        self.data = partition_data
        self.shuffle = shuffle

        self.rand = random.Random()
        self.rand.seed(a=seed)

        # Regions are used to provide indexed access across all utterances
        self.regions = self.get_utt_regions()
        self.region_offsets = [x[0] for x in self.regions]

        # Sampling used to access frames
        self.sampling = list(range(len(self)))

        if self.shuffle:
            self.rand.shuffle(self.sampling)

    def __len__(self):
        last_region = self.regions[-1]
        return last_region[0] + last_region[1]

    def __getitem__(self, item):
        index = self.sampling[item]

        # we search the region before the offset is higher than the index.
        region_index = bisect.bisect_right(self.region_offsets, index) - 1
        region = self.regions[region_index]

        offset_within_utterance = index - region[0]
        return [x[offset_within_utterance].astype(np.float32) for x in region[2]]

    def get_utt_regions(self):
        """
        Return the regions of all utterances, assuming all utterances are concatenated.
        A region is defined by offset, length (num-frames) and
        a list of references to the utterance datasets in the containers.

        Returns:
            list: List of with a tuple for every utterances containing the region info.
        """

        regions = []
        current_offset = 0

        for utt_idx, utt_data in zip(self.data.info.utt_ids, self.data.utt_data):
            offset = current_offset

            lengths = []
            refs = []

            for part in utt_data:
                lengths.append(part.shape[0])
                refs.append(part)

            if len(set(lengths)) != 1:
                raise ValueError('Utterance {} has not the same number of frames in all containers!'.format(utt_idx))

            region = (offset, lengths[0], refs)
            regions.append(region)

            # Sets the offset for the next utterances
            current_offset += lengths[0]

        return regions
