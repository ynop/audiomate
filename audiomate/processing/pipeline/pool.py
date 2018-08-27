import numpy as np

from . import base


class AvgPool(base.Computation):
    """
    Average a given number of sequential frames into a single frame.
    If at the end of a stream just the remaining frames are used, no matter how many there are left.

    Args:
        size (float): The maximum number of frames to pool by taking the mean.
    """

    def __init__(self, size, parent=None, name=None):
        super(AvgPool, self).__init__(min_frames=size, parent=parent, name=name)

        self.size = size
        self.rest = None

    def compute(self, chunk, sampling_rate, corpus=None, utterance=None):

        # Merge incoming plus rest from previous chunk
        all = chunk.data

        if self.rest is not None:
            all = np.vstack([self.rest, all])

        # Only use a multiple of self.size
        num_rest = all.shape[0] % self.size

        if num_rest > 0:
            self.rest = all[-num_rest:, :]
            all = all[:-num_rest, :]
        else:
            self.rest = None

        # Average
        all_mean = np.mean(all.reshape(-1, self.size, all.shape[1]), axis=1)

        if chunk.is_last and self.rest is not None:
            rest_mean = np.mean(self.rest, axis=0)
            all_mean = np.vstack([all_mean, rest_mean])

        return all_mean
