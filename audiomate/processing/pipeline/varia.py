import librosa

from . import base


class Delta(base.Computation):
    """
    Compute delta features.

    See http://librosa.github.io/librosa/generated/librosa.feature.delta.html
    """

    def __init__(self, width=9, order=1, axis=0, mode='interp', parent=None, name=None):
        needed_context = int(width / 2.0)

        super(Delta, self).__init__(parent=parent, name=name,
                                    min_frames=needed_context + 1,
                                    left_context=needed_context, right_context=needed_context)

        self.width = width
        self.order = order
        self.axis = axis
        self.mode = mode

    def compute(self, chunk, sampling_rate, corpus=None, utterance=None):
        axis = len(chunk.data.shape) - self.axis - 1
        output = librosa.feature.delta(chunk.data.T, width=self.width, order=self.order, axis=axis, mode=self.mode).T

        return output[chunk.left_context:chunk.data.shape[0] - chunk.right_context]
