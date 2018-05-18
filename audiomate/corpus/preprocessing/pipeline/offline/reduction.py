import numpy as np

from . import base


class Stack(base.OfflineReduction):
    """
    Stack the features. All input matrices have to be of the same length (same number of frames).
    """

    def compute(self, frames, sampling_rate, corpus=None, utterance=None):
        return np.hstack(frames)
