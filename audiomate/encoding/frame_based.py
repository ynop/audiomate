import numpy as np

from . import base
from audiomate.utils import misc


class FrameHotEncoder(base.Encoder):
    """
    The FrameHotEncoder is used to encode the labels per frame.
    It creates a matrix with dimension num-frames x len(labels).
    The vector (2nd dim) has an entry for every label in the passed labels-list.
    If the sequence contains a given label within a frame it is set to 1.

    Arguments:
        labels (list): List of labels (str) which should be included in the vector representation.
        label_list_idx (str): The name of the label-list to use for encoding.
                              Only labels of this label-list are considered.
        frame_settings (FrameSettings): Frame settings to use.
        sr (int): The sampling rate used, if None it is assumed the native sampling rate from the file is used.

    Example:
        >>> from audiomate import annotations
        >>> from audiomate.utils import units import
        >>>
        >>> ll = annotations.LabelList(idx='test', labels=[
        >>>     annotations.Label('music', 0, 2),
        >>>     annotations.Label('speech', 2, 5),
        >>>     annotations.Label('noise', 4, 6),
        >>>     annotations.Label('music', 6, 8)
        >>> ])
        >>> utt.set_label_list(ll)
        >>>
        >>> labels = ['speech', 'music', 'noise']
        >>> fs = units.FrameSettings(16000, 16000)
        >>> encoder = FrameHotEncoder(labels, 'test', frame_settings=fs, sr=16000)
        >>> encoder.encode_utterance(utt)
        array([
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0]
        ])

    """

    def __init__(self, labels, label_list_idx, frame_settings, sr=None):
        self.labels = labels
        self.label_list_idx = label_list_idx
        self.frame_settings = frame_settings
        self.sr = sr

    def encode_utterance(self, utterance, corpus=None):
        sr = self.sr or utterance.sampling_rate
        num_samples = utterance.num_samples(sr=sr)
        num_frames = self.frame_settings.num_frames(num_samples)

        mat = np.zeros((num_frames, len(self.labels)))

        if self.label_list_idx not in utterance.label_lists:
            raise ValueError('Utterance {} has no label-list with idx {}'.format(utterance.idx, self.label_list_idx))

        label_list = utterance.label_lists[self.label_list_idx]

        for label in label_list:
            if label.value in self.labels:
                start, end = self.frame_settings.time_range_to_frame_range(label.start, label.end, sr)

                # If label ends at the end of the utterance
                if label.end < 0:
                    end = num_frames

                mat[start:end, self.labels.index(label.value)] = 1

        return mat


class FrameOrdinalEncoder(base.Encoder):
    """
    The FrameOrdinalEncoder is used to encode the labels per frame.
    It creates a vector with length num-frames.
    For every frame sets the index of the label that is present for that frame.
    If multiple labels are present the longest within the frame.
    If multiple labels have the same length the smaller index is selected, hence
    the passed `labels` list acts as a priority.

    Arguments:
        labels (list): List of labels (str) which should be included in the vector representation.
        label_list_idx (str): The name of the label-list to use for encoding.
                              Only labels of this label-list are considered.
        frame_settings (FrameSettings): Frame settings to use.
        sr (int): The sampling rate used, if None it is assumed the native sampling rate from the file is used.

    Example:
        >>> from audiomate import annotations
        >>> from audiomate.utils import units import
        >>>
        >>> ll = annotations.LabelList(idx='test', labels=[
        >>>     annotations.Label('music', 0, 2),
        >>>     annotations.Label('speech', 2, 5),
        >>>     annotations.Label('noise', 4, 6),
        >>>     annotations.Label('music', 6, 8)
        >>> ])
        >>> utt.set_label_list(ll)
        >>>
        >>> labels = ['speech', 'music', 'noise']
        >>> fs = units.FrameSettings(16000, 16000)
        >>> encoder = FrameOrdinalEncoder(labels, 'test', frame_settings=fs)
        >>> encoder.encode_utterance(utt)
        array([1,1,0,0,0,2,1,1])
    """

    def __init__(self, labels, label_list_idx, frame_settings, sr=None):
        self.labels = labels
        self.label_list_idx = label_list_idx
        self.frame_settings = frame_settings
        self.sr = sr

    def encode_utterance(self, utterance, corpus=None):
        sr = self.sr or utterance.sampling_rate
        num_samples = utterance.num_samples(sr=sr)
        num_frames = self.frame_settings.num_frames(num_samples)

        mat = np.zeros((num_frames, len(self.labels)))

        if self.label_list_idx not in utterance.label_lists:
            raise ValueError('Utterance {} has no label-list with idx {}'.format(utterance.idx, self.label_list_idx))

        label_list = utterance.label_lists[self.label_list_idx]

        for label in label_list:
            if label.value in self.labels:
                start, end = self.frame_settings.time_range_to_frame_range(label.start, label.end, sr)

                # If label ends at the end of the utterance
                if label.end < 0:
                    end = num_frames

                label_index = self.labels.index(label.value)

                for frame_index in range(start, min(end, num_frames)):
                    frame_start, frame_end = self.frame_settings.frame_to_seconds(frame_index, sr=sr)
                    overlap = misc.length_of_overlap(frame_start, frame_end, label.start, label.end)

                    mat[frame_index, label_index] = overlap

        return np.argmax(mat, axis=1)
