import numpy as np

from . import base


class Stack(base.Reduction):
    """
    Stack the features from multiple inputs.
    All input matrices have to be of the same length (same number of frames).
    """

    def compute(self, chunk, sampling_rate, corpus=None, utterance=None):
        return np.hstack(chunk.data)
