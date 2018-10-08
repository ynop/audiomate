import bisect
import math

import numpy as np

import audiomate
from audiomate.corpus import assets


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

    def __init__(self, corpus_or_utt_ids, containers):
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

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class FrameDataset(Dataset):
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
        >>> container_inputs = assets.FeatureContainer('/path/to/features.hdf5')
        >>> container_outputs = assets.Container('/path/to/targets.hdf5')
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
        super(FrameDataset, self).__init__(corpus_or_utt_ids, containers)

        self.regions = self.get_utt_regions()
        self.region_offsets = [x[0] for x in self.regions]

    def __len__(self):
        last_region = self.regions[-1]
        return last_region[0] + last_region[1]

    def __getitem__(self, item):
        # we search the region before the offset is higher than the item.
        region_index = bisect.bisect_right(self.region_offsets, item) - 1
        region = self.regions[region_index]

        offset_within_utterance = item - region[0]
        return [x[offset_within_utterance].astype(np.float32) for x in region[2]]

    def get_utt_regions(self):
        """
        Return the regions of all utterances, assuming all utterances are concatenated.
        It is assumed that the utterances are sorted in ascending order for concatenation.

        A region is defined by offset, length (num-frames) and
        a list of references to the utterance datasets in the containers.

        Returns:
            list: List of with a tuple for every utterances containing the region info.
        """

        regions = []
        current_offset = 0

        for utt_idx in sorted(self.utt_ids):
            offset = current_offset

            lengths = []
            refs = []

            for container in self.containers:
                lengths.append(container.get(utt_idx).shape[0])
                refs.append(container.get(utt_idx, mem_map=True))

            if len(set(lengths)) != 1:
                raise ValueError('Utterance {} has not the same number of frames in all containers!'.format(utt_idx))

            region = (offset, lengths[0], refs)
            regions.append(region)

            # Sets the offset for the next utterances
            current_offset += lengths[0]

        return regions


class MultiFrameDataset(FrameDataset):
    """
    A dataset wrapping chunks of frames of a corpus. A single sample represents a chunk of frames.

    A chunk doesn't overlap an utterances boundaries. So if the utterance length is not divisible by the chunk length,
    the last chunk of an utterance may be smaller than the chunk size.

    Args:
        corpus_or_utt_ids (Corpus, list): Either a corpus or a list of utterances.
                                          This defines which utterances are considered for iterating.
        containers (list, Container): A single container or a list of containers.
        frames_per_chunk (int): Number of subsequent frames in a single sample.

    Note:
        For a multi-frame dataset it is expected that every container contains exactly one value/vector for every frame.
        So the first dimension of every array in every container have to match.

    Example:
        >>> corpus = audiomate.Corpus.load('/path/to/corpus')
        >>> container_inputs = assets.FeatureContainer('/path/to/features.hdf5')
        >>> container_outputs = assets.Container('/path/to/targets.hdf5')
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
                   [0.59310933, 0.8565602 , 0.95468696, 0.07933512]])
            array([[0.0, 1.0], [0.0, 1.0],[0.0, 1.0],[0.0, 1.0], [0.0, 1.0]])
        )
    """

    def __init__(self, corpus_or_utt_ids, containers, frames_per_chunk):
        super(MultiFrameDataset, self).__init__(corpus_or_utt_ids, containers)

        self.frames_per_chunk = frames_per_chunk

        self.chunked_regions = self.get_chunked_utt_regions()
        self.chunked_offsets = [x[0] for x in self.chunked_regions]

    def __len__(self):
        last_region = self.chunked_regions[-1]
        return last_region[0] + last_region[1]

    def __getitem__(self, item):
        # we search the region before the offset is higher than the item.
        region_index = bisect.bisect_right(self.chunked_offsets, item) - 1
        region = self.chunked_regions[region_index]

        offset_within_utterance = (item - region[0]) * self.frames_per_chunk
        to = offset_within_utterance + self.frames_per_chunk
        return [x[offset_within_utterance:to].astype(np.float32) for x in region[2]]

    def get_chunked_utt_regions(self):
        """
        From the regions defined for the single-frame case, extract regions for chunked access.
        """
        chunked_regions = []
        chunked_offset = 0

        for offset, length, references in self.regions:
            chunked_length = math.ceil(length / float(self.frames_per_chunk))
            chunked_regions.append((chunked_offset, chunked_length, references))

            chunked_offset += chunked_length

        return chunked_regions
