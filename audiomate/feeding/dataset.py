import bisect
import math

import numpy as np

import audiomate
from audiomate import containers
from audiomate import processing
from . import iterator


class Dataset(object):
    """
    An abstract class representing a dataset. A dataset provides indexable access to data.
    An implementation of a concrete dataset should override the methods ``__len__`` and ``__getitem``.

    A sample returned from a dataset is a tuple containing the data for this sample from every container.
    The data from different containers is ordered in the way the containers were passed to the Dataset.

    Args:
        corpus_or_utt_ids (Corpus, list): Either a corpus or a list of utterances.
                                          This defines which utterances are considered for iterating.
        containers (list, Container): A single container or a list of containers.
    """

    def __init__(self, corpus_or_utt_ids, feature_containers):
        if isinstance(corpus_or_utt_ids, audiomate.corpus.CorpusView):
            self.utt_ids = sorted(list(corpus_or_utt_ids.utterances.keys()))
        else:
            self.utt_ids = sorted(corpus_or_utt_ids)

        if isinstance(feature_containers, containers.Container):
            self.containers = [feature_containers]
        else:
            self.containers = feature_containers

        if len(self.containers) == 0:
            raise ValueError('At least one container has to be provided!')

        for cnt in self.containers:
            if not Dataset.container_has_utterances(cnt, self.utt_ids):
                raise ValueError('Container {} does not contain all necessary utterances'.format(cnt.path))

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def container_has_utterances(container, keys):
        container_keys = set(container.keys())
        return container_keys.issuperset(keys)


class UtteranceDataset(Dataset):
    """
    A dataset wrapping data of complete utterances. Assuming two containers are given,
    a single sample contains all the data for an utterance from both containers.

    Args:
        corpus_or_utt_ids (Corpus, list): Either a corpus or a list of utterances.
                                          This defines which utterances are considered for iterating.
        source_containers (list, Container): A single container or a list of containers.
        return_length (bool): If True, the length of the data from every container is returned as well.
                              The length is appended after the data of every container
                              (e.g. [container1-data, container1-length, container2-data, container2-length])
        pad (bool): If True, samples that are shorter than the longest sample in the dataset are padded with 0.
                    The padding is done for each container individually.
                    (e.g. container1-data can be shorter than container2-data)
                    If padding is enabled, the lengths are always returned ``return_length = True``.
        transform (list, Processor): A single processor or a list of processors.
                                     Has to be the same number as the number of containers.
                                     If for some containers no transform should be applied,
                                     pass ``None``.

    Examples:

        >>> corpus = audiomate.Corpus.load('/path/to/corpus')
        >>> container_inputs = containers.FeatureContainer('/path/to/features.hdf5')
        >>> container_outputs = containers.Container('/path/to/targets.hdf5')
        >>>
        >>> ds = UtteranceDataset(corpus, [container_inputs, container_outputs])
        >>> len(ds) # Number of utterances/samples in the dataset
        114
        >>> ds[20] # Utterance/Sample (inputs, inputs-len, outputs, outputs-len) with index 20
        (
            array([[0.72991909, 0.20258683, 0.30574747, 0.53783217],
                   [0.38875413, 0.83611128, 0.49054591, 0.15710017],
                   [0.35153358, 0.40051009, 0.93647765, 0.29589257],
                   [0.97465772, 0.80160451, 0.81871436, 0.4892925 ],
                   [0.59310933, 0.8565602 , 0.95468696, 0.07933512]]),
            5,
            array([0, 9, 10, 4]),
            4
        )

    With ``pad=True`` the data is padded on the outermost dimension
    with zeros to match the length of the longest sequence/utterance.
    In the following it is assumed the longest sequence in the inputs is ``8`` and in the outputs ``6``.

        >>> corpus = audiomate.Corpus.load('/path/to/corpus')
        >>> container_inputs = containers.FeatureContainer('/path/to/features.hdf5')
        >>> container_outputs = containers.Container('/path/to/targets.hdf5')
        >>>
        >>> ds = UtteranceDataset(corpus, [container_inputs, container_outputs], pad=True)
        >>> len(ds) # Number of utterances/samples in the dataset
        114
        >>> ds[20] # Utterance/Sample (inputs, inputs-len, outputs, outputs-len) with index 20
        (
            array([[0.72991909, 0.20258683, 0.30574747, 0.53783217],
                   [0.38875413, 0.83611128, 0.49054591, 0.15710017],
                   [0.35153358, 0.40051009, 0.93647765, 0.29589257],
                   [0.97465772, 0.80160451, 0.81871436, 0.4892925 ],
                   [0.59310933, 0.8565602 , 0.95468696, 0.07933512],
                   [0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]
            ]),
            5,
            array([0, 9, 10, 4, 0, 0]),
            4
        )

    """

    def __init__(self, corpus_or_utt_ids, source_containers, return_length=True, pad=False, transform=None):
        super(UtteranceDataset, self).__init__(corpus_or_utt_ids, source_containers)

        self.pad = pad

        if transform is None:
            self.transform = []
        else:
            if isinstance(transform, processing.Processor):
                self.transform = [transform]
            else:
                self.transform = transform

            if len(self.transform) != len(self.containers):
                raise ValueError(
                    ('Number of transforms does not match number of containers!'
                     'Use None for containers without transform!')
                )

            for index, container_transform in enumerate(self.transform):
                if container_transform is not None and \
                        not isinstance(self.containers[index], containers.FeatureContainer):
                    raise ValueError(
                        ('Transforms can only be applied to'
                         'Feature-Containers (Transform at index {})').format(index))

        if self.pad:
            self.return_length = True
        else:
            self.return_length = return_length

        self.pad_lengths = self.longest_utterances_per_container()

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, item):
        utt_idx = self.utt_ids[item]

        sample = []

        for index, cnt in enumerate(self.containers):
            data = cnt.get(utt_idx, mem_map=False)

            if index < len(self.transform) and self.transform[index] is not None:
                data = self.transform[index].process_frames(
                    data,
                    cnt.sampling_rate,
                    offset=0,
                    last=True
                )

            size = data.shape[0]

            required_padded_length = self.pad_lengths[index]

            if self.pad and size < required_padded_length:
                pad_width = [(0, required_padded_length - size)]

                for i in range(1, len(data.shape)):
                    pad_width.append((0, 0))

                data = np.pad(data, pad_width, mode='constant', constant_values=0)

            sample.append(data)

            if self.return_length:
                sample.append(size)

        return sample

    def longest_utterances_per_container(self):
        """ Return a tuple/list containing the length of the longest utterance of ever container. """
        lengths = []

        for cnt in self.containers:
            longest_in_container = 0
            for utt_idx in self.utt_ids:
                utt_length = cnt._file[utt_idx].shape[0]
                longest_in_container = max(utt_length, longest_in_container)

            lengths.append(longest_in_container)

        return lengths


class MultiFrameDataset(Dataset):
    """
    A dataset wrapping chunks of frames of a corpus. A single sample represents a chunk of frames.

    A chunk doesn't overlap an utterances boundaries. So if the utterance length is not divisible by the chunk length,
    the last chunk of an utterance may be smaller than the chunk size.

    Args:
        corpus_or_utt_ids (Corpus, list): Either a corpus or a list of utterances.
                                          This defines which utterances are considered for iterating.
        containers (list, Container): A single container or a list of containers.
        frames_per_chunk (int): Number of subsequent frames in a single sample.
        return_length (bool): If True, the length of the chunk is returned as well. (default ``False``)
                              The length is appended to tuple as the last element.
                              (e.g. [container1-data, container2-data, length])
        pad (bool): If True, samples that are shorter are padded with zeros to match ``frames_per_chunk``.
                    If padding is enabled, the lengths are always returned ``return_length = True``.

    Note:
        For a multi-frame dataset it is expected that every container contains exactly one value/vector for every frame.
        So the first dimension of every array in every container have to match.

    Examples:

        >>> corpus = audiomate.Corpus.load('/path/to/corpus')
        >>> container_inputs = containers.FeatureContainer('/path/to/features.hdf5')
        >>> container_outputs = containers.Container('/path/to/targets.hdf5')
        >>>
        >>> ds = MultiFrameDataset(corpus, [container_inputs, container_outputs], 5)
        >>> len(ds) # Number of chunks in the dataset
        355
        >>> ds[20] # Chunk (inputs, outputs) with index 20
        (
            array([[0.72991909, 0.20258683, 0.30574747, 0.53783217],
                   [0.38875413, 0.83611128, 0.49054591, 0.15710017],
                   [0.35153358, 0.40051009, 0.93647765, 0.29589257],
                   [0.97465772, 0.80160451, 0.81871436, 0.4892925 ],
                   [0.59310933, 0.8565602 , 0.95468696, 0.07933512]]),
            array([[0.0, 1.0], [0.0, 1.0],[0.0, 1.0],[0.0, 1.0], [0.0, 1.0]])
        )

    If the length should be returned, pass ``True`` to ``return_length``
    (Except for chunks at the of utterances the length will be equal to ``frames_per_chunk``.)

        >>> corpus = audiomate.Corpus.load('/path/to/corpus')
        >>> container_inputs = containers.FeatureContainer('/path/to/features.hdf5')
        >>> container_outputs = containers.Container('/path/to/targets.hdf5')
        >>>
        >>> ds = MultiFrameDataset(corpus, [container_inputs, container_outputs], 5)
        >>> len(ds) # Number of chunks in the dataset
        355
        >>> ds[20] # Chunk (inputs, outputs) with index 20
        (
            array([[0.72991909, 0.20258683, 0.30574747, 0.53783217],
                   [0.38875413, 0.83611128, 0.49054591, 0.15710017],
                   [0.35153358, 0.40051009, 0.93647765, 0.29589257],
                   [0.97465772, 0.80160451, 0.81871436, 0.4892925 ],
                   [0.59310933, 0.8565602 , 0.95468696, 0.07933512]]),
            array([[0.0, 1.0], [0.0, 1.0],[0.0, 1.0],[0.0, 1.0], [0.0, 1.0]]),
            5
        )
    """

    def __init__(self, corpus_or_utt_ids, containers, frames_per_chunk, return_length=False, pad=False):
        super(MultiFrameDataset, self).__init__(corpus_or_utt_ids, containers)

        if frames_per_chunk < 1:
            raise ValueError('Chunk-size has to be at least 1!')

        self.frames_per_chunk = frames_per_chunk
        self.pad = pad

        if self.pad:
            self.return_length = True
        else:
            self.return_length = return_length

        self.regions = self.get_utt_regions()
        self.region_offsets = [x[0] for x in self.regions]

    def __len__(self):
        last_region = self.regions[-1]
        return last_region[0] + last_region[1]

    def __getitem__(self, item):
        # we search the region before the offset is higher than the item.
        region_index = bisect.bisect_right(self.region_offsets, item) - 1
        region = self.regions[region_index]

        frame_offset = (item - region[0]) * self.frames_per_chunk
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

    def partitioned_iterator(self, partition_size, shuffle=True, seed=None):
        """
        Return a partitioning :class:`audiomate.feeding.MultiFrameIterator` for the dataset.

        Args:
            partition_size (str): Size of the partitions in bytes. The units ``k`` (kibibytes), ``m``
                                  (mebibytes) and ``g`` (gibibytes) are supported, i.e. a ``partition_size``
                                  of ``1g`` equates :math:`2^{30}` bytes.
            shuffle (bool): Indicates whether the data should be returned in
                            random order (``True``) or not (``False``).
            seed (int): Seed to be used for the random number generator.

        Returns:
            MultiFrameIterator: A partition iterator over the dataset.
        """
        return iterator.MultiFrameIterator(self.utt_ids, self.containers, partition_size, self.frames_per_chunk,
                                           return_length=self.return_length, pad=self.pad, shuffle=shuffle, seed=seed)

    def get_utt_regions(self):
        """
        Return the regions of all utterances, assuming all utterances are concatenated.
        It is assumed that the utterances are sorted in ascending order for concatenation.

        A region is defined by offset (in chunks), length (num-chunks) and
        a list of references to the utterance datasets in the containers.

        Returns:
            list: List of with a tuple for every utterances containing the region info.
        """

        regions = []
        current_offset = 0

        for utt_idx in sorted(self.utt_ids):
            offset = current_offset

            num_frames = []
            refs = []

            for cnt in self.containers:
                num_frames.append(cnt.get(utt_idx).shape[0])
                refs.append(cnt.get(utt_idx, mem_map=True))

            if len(set(num_frames)) != 1:
                raise ValueError('Utterance {} has not the same number of frames in all containers!'.format(utt_idx))

            num_chunks = math.ceil(num_frames[0] / float(self.frames_per_chunk))

            region = (offset, num_chunks, refs)
            regions.append(region)

            # Sets the offset for the next utterances
            current_offset += num_chunks

        return regions


class FrameDataset(MultiFrameDataset):
    """
    A dataset wrapping frames of a corpus. A single sample represents a single frame.

    Args:
        corpus_or_utt_ids (Corpus, list): Either a corpus or a list of utterances.
                                          This defines which utterances are considered for iterating.
        containers (list, Container): A single container or a list of containers.

    Note:
        For a frame dataset it is expected that every container contains exactly one value/vector for every frame.
        So the first dimension of every array in every container have to match.

    Example:
        >>> corpus = audiomate.Corpus.load('/path/to/corpus')
        >>> container_inputs = containers.FeatureContainer('/path/to/features.hdf5')
        >>> container_outputs = containers.Container('/path/to/targets.hdf5')
        >>>
        >>> ds = FrameDataset(corpus, [container_inputs, container_outputs])
        >>> len(ds) # Number of frames in the dataset
        2938
        >>> ds[293] # Frame (inputs, outputs) with index 293
        (
            array([0.58843831, 0.18128443, 0.19718328, 0.25284105]),
            array([0.0, 1.0])
        )
    """

    def __init__(self, corpus_or_utt_ids, containers):
        super(FrameDataset, self).__init__(corpus_or_utt_ids, containers, 1, return_length=False)

    def __getitem__(self, item):
        data = super(FrameDataset, self).__getitem__(item)

        # We have to remove the outermost dimension, which is 1 for chunk-size of 1 frame
        return [x[0] for x in data]

    def partitioned_iterator(self, partition_size, shuffle=True, seed=None):
        """
        Return a partitioning :class:`audiomate.feeding.FrameIterator` for the dataset.

        Args:
            partition_size (str): Size of the partitions in bytes. The units ``k`` (kibibytes), ``m``
                                  (mebibytes) and ``g`` (gibibytes) are supported, i.e. a ``partition_size``
                                  of ``1g`` equates :math:`2^{30}` bytes.
            shuffle (bool): Indicates whether the data should be returned in
                            random order (``True``) or not (``False``).
            seed (int): Seed to be used for the random number generator.

        Returns:
            FrameIterator: A partition iterator over the dataset.
        """
        return iterator.FrameIterator(self.utt_ids, self.containers, partition_size, shuffle=shuffle, seed=seed)
