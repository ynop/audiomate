import bisect

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

    Note:
        For a frame dataset it is expected that every container contains exactly one value/vector for every frame.
        So the first dimension of every array in every container have to match.
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
