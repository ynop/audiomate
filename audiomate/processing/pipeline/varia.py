import librosa

from . import base


class Delta(base.Computation):
    """
    Compute delta features.

    See http://librosa.github.io/librosa/generated/librosa.feature.delta.html
    """

    def __init__(self, width=9, order=1, axis=-1, mode='interp', parent=None, name=None):
        super(Delta, self).__init__(parent=parent, name=name)

        self.width = width
        self.order = order
        self.axis = axis
        self.mode = mode

    def compute(self, data, sampling_rate, first_frame_index=None, last=False, corpus=None, utterance=None):
        return librosa.feature.delta(data.T, width=self.width, order=self.order, axis=self.axis, mode=self.mode).T
