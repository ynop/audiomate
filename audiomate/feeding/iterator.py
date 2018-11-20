import bisect
import math
import random

import numpy as np

import audiomate
from audiomate import containers
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

    def __init__(self, corpus_or_utt_ids, feature_containers, shuffle=True, seed=None):
        if isinstance(corpus_or_utt_ids, audiomate.corpus.CorpusView):
            self.utt_ids = list(corpus_or_utt_ids.utterances.keys())
        else:
            self.utt_ids = corpus_or_utt_ids

        if isinstance(feature_containers, containers.Container):
            self.containers = [feature_containers]
        else:
            self.containers = feature_containers

        if len(self.containers) == 0:
            raise ValueError('At least one container has to be provided!')

        self.shuffle = shuffle

        self.rand = random.Random()
        self.rand.seed(a=seed)

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class MultiFrameIterator(DataIterator):
    """
    A data-iterator wrapping chunks of subsequent frames of a corpus.
    A single sample represents a chunk of frames.

    Args:
        corpus_or_utt_ids (Corpus, list): Either a corpus or a list of utterances.
                                          This defines which utterances are considered for iterating.
        container (list, Container): A single container or a list of containers.
        partition_size (str): Size of the partitions in bytes. The units ``k`` (kibibytes), ``m`` (mebibytes) and ``g``
                              (gibibytes) are supported, i.e. a ``partition_size`` of ``1g`` equates :math:`2^{30}`
                              bytes.
        frames_per_chunk (int): Number of subsequent frames in a single sample.
        return_length (bool): If True, the length of the chunk is returned as well. (default ``False``)
                              The length is appended to tuple as the last element.
                              (e.g. [container1-data, container2-data, length])
        pad (bool): If True, samples that are shorter are padded with zeros to match ``frames_per_chunk``.
                    If padding is enabled, the lengths are always returned ``return_length = True``.
        shuffle (bool): Indicates whether the data should be returned in
                        random order (``True``) or not (``False``).
        seed (int): Seed to be used for the random number generator.

    Note:
        For a MultiFrameIterator it is expected that every container contains exactly one value/vector for every frame.
        So the first dimension (outermost) of every array in every container have to match.

    Example:
        >>> corpus = audiomate.Corpus.load('/path/to/corpus')
        >>> container_inputs = containers.FeatureContainer('/path/to/features.hdf5')
        >>> container_outputs = containers.Container('/path/to/targets.hdf5')
        >>>
        >>> ds = MultiFrameIterator(corpus, [container_inputs, container_outputs], '1G', 5, shuffle=True, seed=23)
        >>> next(ds) # Next Chunk (inputs, outputs)
        (
            array([[0.72991909, 0.20258683, 0.30574747, 0.53783217],
                   [0.38875413, 0.83611128, 0.49054591, 0.15710017],
                   [0.35153358, 0.40051009, 0.93647765, 0.29589257],
                   [0.97465772, 0.80160451, 0.81871436, 0.4892925 ],
                   [0.59310933, 0.8565602 , 0.95468696, 0.07933512]])
            array([[0.0, 1.0], [0.0, 1.0],[0.0, 1.0],[0.0, 1.0], [0.0, 1.0]])
        )
    """

    def __init__(self, corpus_or_utt_ids, containers, partition_size, frames_per_chunk, return_length=False,
                 pad=False, shuffle=True, seed=None):
        super(MultiFrameIterator, self).__init__(corpus_or_utt_ids, containers, shuffle=shuffle, seed=seed)

        self.partition_size = partition_size
        self.frames_per_chunk = frames_per_chunk
        self.pad = pad

        if self.pad:
            self.return_length = True
        else:
            self.return_length = return_length

        self.loader = None
        self.current_partition = None
        self.current_partition_index = -1
        self.current_chunk_index = 0

        self.loader = partitioning.PartitioningContainerLoader(self.utt_ids,
                                                               self.containers,
                                                               self.partition_size,
                                                               shuffle=self.shuffle,
                                                               seed=self.rand.random())

    def __iter__(self):
        self.current_partition = None
        self.current_partition_index = -1
        self.current_chunk_index = 0

        self.loader.reload()

        return self

    def __next__(self):
        if self.current_partition is None or self.current_chunk_index >= len(self.current_partition):
            self.current_partition_index += 1
            self.current_chunk_index = 0

            if self.current_partition_index < len(self.loader.partitions):
                partition_data = self.loader.load_partition_data(self.current_partition_index)
                self.current_partition = MultiFramePartitionData(partition_data,
                                                                 self.frames_per_chunk,
                                                                 return_length=self.return_length,
                                                                 pad=self.pad,
                                                                 shuffle=self.shuffle,
                                                                 seed=self.rand.random())
            else:
                raise StopIteration

        next_chunk = self.current_partition[self.current_chunk_index]
        self.current_chunk_index += 1

        return next_chunk


class FrameIterator(MultiFrameIterator):
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

    Example:
        >>> corpus = audiomate.Corpus.load('/path/to/corpus')
        >>> container_inputs = containers.FeatureContainer('/path/to/features.hdf5')
        >>> container_outputs = containers.Container('/path/to/targets.hdf5')
        >>>
        >>> ds = FrameIterator(corpus, [container_inputs, container_outputs], '1G', shuffle=True, seed=23)
        >>> next(ds) # Next Frame (inputs, outputs)
        (
            array([0.58843831, 0.18128443, 0.19718328, 0.25284105]),
            array([0.0, 1.0])
        )
    """

    def __init__(self, corpus_or_utt_ids, containers, partition_size, shuffle=True, seed=None):
        super(FrameIterator, self).__init__(corpus_or_utt_ids, containers, partition_size, 1,
                                            return_length=False, shuffle=shuffle, seed=seed)

    def __next__(self):
        data = super(FrameIterator, self).__next__()

        # We have to remove the outermost dimension, which is 1 for chunk-size of 1 frame
        return [x[0] for x in data]


class MultiFramePartitionData(object):
    """
    Wrapper for PartitionData to access chunks of frames via indexes.

    Args:
        partition_data (PartitionData): The loaded partition-data.
        frames_per_chunk (int): Number of subsequent frames in a chunk.
        return_length (bool): If True, the length of the chunk is returned as well. (default ``False``)
                              The length is appended to tuple as the last element.
                              (e.g. [container1-data, container2-data, length])
        pad (bool): If True, samples that are shorter are padded with zeros to match ``frames_per_chunk``.
                    If padding is enabled, the lengths are always returned ``return_length = True``.
        shuffle (bool): If True the frames are shuffled randomly for access.
        seed (int): The seed to use for shuffling.
    """

    def __init__(self, partition_data, frames_per_chunk, return_length=False, pad=False, shuffle=True, seed=None):
        if frames_per_chunk < 1:
            raise ValueError('Number of frames per chunk has to higher than 0.')

        self.data = partition_data
        self.frames_per_chunk = frames_per_chunk
        self.pad = pad

        if self.pad:
            self.return_length = True
        else:
            self.return_length = return_length

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

        frame_offset = (index - region[0]) * self.frames_per_chunk
        frame_end = frame_offset + self.frames_per_chunk

        data = [x[frame_offset:frame_end].astype(np.float32) for x in region[2]]
        size = data[0].shape[0]

        if self.pad and size < self.frames_per_chunk:
            padded_data = []
            for x in data:
                # Only pad the outermost (first) dimension
                pad_widths = [(0, 0)] * (len(x.shape) - 1)
                pad_widths.insert(0, (0, self.frames_per_chunk - size))
                padded_x = np.pad(x, pad_widths, mode='constant', constant_values=0)
                padded_data.append(padded_x)

            data = padded_data

        if self.return_length:
            data.append(size)

        return data

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

            num_frames = []
            refs = []

            for part in utt_data:
                num_frames.append(part.shape[0])
                refs.append(part)

            if len(set(num_frames)) != 1:
                raise ValueError('Utterance {} has not the same number of frames in all containers!'.format(utt_idx))

            num_chunks = math.ceil(num_frames[0] / float(self.frames_per_chunk))

            region = (offset, num_chunks, refs)
            regions.append(region)

            # Sets the offset for the next utterances
            current_offset += num_chunks

        return regions
