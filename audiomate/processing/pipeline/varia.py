import librosa
import numpy as np

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


class AddContext(base.Computation):
    """
    For every frame add context frames from left or/and right.
    For frames at the beginning and end of a sequence, where no context is available, zeros are used.

    Args:
        left_frames (int): Number of previous frames to prepend to a frame.
        right_frames (int): Number of subsequent frames to append to a frame.

    Example:
        >>> input = np.array([
        >>>     [1,2,3],
        >>>     [4,5,6],
        >>>     [7,8,9]
        >>> ])
        >>> chunk = Chunk(input, offset=0, is_last=True)
        >>> AddContext(left_frames=1, right_frames=1).compute(chunk, 16000)
        array([[0, 0, 0, 1, 2, 3, 4, 5, 6],
               [1, 2, 3, 4, 5, 6, 7, 8, 9],
               [4, 5, 6, 7, 8, 9, 0, 0, 0]])
    """

    def __init__(self, left_frames, right_frames, parent=None, name=None):
        super(AddContext, self).__init__(parent=parent, name=name, min_frames=1,
                                         left_context=left_frames, right_context=right_frames)

        self.left_frames = left_frames
        self.right_frames = right_frames

    def compute(self, chunk, sampling_rate, corpus=None, utterance=None):
        context = []

        for shift in range(self.left_frames, 0, -1):
            shift_context = chunk.data[:chunk.data.shape[0] - shift]
            pad_widths = [[0, 0] for __ in range(len(shift_context.shape))]
            pad_widths[0][0] = shift
            shift_context = np.pad(shift_context, pad_widths, mode='constant', constant_values=0)
            context.append(shift_context)

        context.append(chunk.data)

        for i in range(self.right_frames):
            shift = i + 1
            shift_context = chunk.data[shift:]
            pad_widths = [[0, 0] for __ in range(len(shift_context.shape))]
            pad_widths[0][1] = shift
            shift_context = np.pad(shift_context, pad_widths, mode='constant', constant_values=0)
            context.append(shift_context)

        stacked = np.hstack(context)
        return stacked[chunk.left_context:chunk.data.shape[0] - chunk.right_context]
