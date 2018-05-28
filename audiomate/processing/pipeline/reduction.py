import numpy as np

from . import base


class Stack(base.Reduction):
    """
    Stack the features. All input matrices have to be of the same length (same number of frames).
    """

    def compute(self, data, sampling_rate, first_frame_index=None, last=False, corpus=None, utterance=None):
        return np.hstack(data)
