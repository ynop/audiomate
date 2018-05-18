"""
This module contains code to convert label-lists to vector representations.
This is for example used when training a classifier and the labels act as target values.
Then they can be converted to a numerical representation using Encoders.
"""

import numpy as np

from audiomate.utils import misc


class Encoder(object):
    """
    An encoder is used to create a numerical vector representation from labels.
    """

    def encode(self, utterance, label_list_idx='default'):
        """
        Encode the given utterance.

        Args:
            utterance (Utterance): The utterance to encode.
            label_list_idx (str): The name of the label-list to use for encoding.
                                  Only labels of this label-list are considered.

        Returns:
            np.ndarray: The array containing the encoded labels. (num_frames x num_labels)
        """
        pass


class FrameOneHotEncoder(Encoder):
    """
    The FrameOneHotEncoder is used to encode the labels per frame.
    It creates a matrix with dimension num-frames x len(labels).
    The vector (2nd dim) has an entry for every label in the passed labels-list.
    If the sequence contains a given label within a frame it is set to 1.

    Arguments:
        labels (list): List of labels (str) which should be included in the vector representation.
        frame_settings (FrameSettings): Frame settings to use.
        sr (int): The sampling rate used, if None it is assumed the native sampling rate from the file is used.

    Example:
        >>> from audiomate.corpus import assets
        >>> from audiomate.utils import units import
        >>> ll = assets.LabelList(labels=[
        >>>     assets.Label('music', 0, 2),
        >>>     assets.Label('speech', 2, 5),
        >>>     assets.Label('noise', 4, 6),
        >>>     assets.Label('music', 6, 8)
        >>> ])
        >>> labels = ['speech', 'music', 'noise']
        >>> fs = units.FrameSettings(16000, 16000)
        >>> encoder = FrameOneHotEncoder(labels, frame_settings=fs)
        >>> encoder.encode(ll)
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

    def __init__(self, labels, frame_settings, sr=None):
        self.labels = labels
        self.frame_settings = frame_settings
        self.sr = sr

    def encode(self, utterance, label_list_idx='default'):
        sr = self.sr or utterance.sampling_rate
        num_samples = utterance.num_samples(sr=sr)
        num_frames = self.frame_settings.num_frames(num_samples)

        mat = np.zeros((num_frames, len(self.labels)))

        if label_list_idx not in utterance.label_lists:
            raise ValueError('Utterance {} has no label-list with idx {}'.format(utterance.idx, label_list_idx))

        label_list = utterance.label_lists[label_list_idx]

        for label in label_list:
            if label.value in self.labels:
                start, end = self.frame_settings.time_range_to_frame_range(label.start, label.end, sr)

                # If label ends at the end of the utterance
                if label.end < 0:
                    end = num_frames

                mat[start:end, self.labels.index(label.value)] = 1

        return mat


class FrameOrdinalEncoder(Encoder):
    """
    The FrameOrdinalEncoder is used to encode the labels per frame.
    It creates a vector with length num-frames.
    For every frame sets the index of the label that is present for that frame.
    If multiple labels are present the longest within the frame.
    If multiple labels have the same length the smaller index is selected, hence
    the passed `labels` list acts as a priority.

    Arguments:
        labels (list): List of labels (str) which should be included in the vector representation.
        frame_settings (FrameSettings): Frame settings to use.
        sr (int): The sampling rate used, if None it is assumed the native sampling rate from the file is used.

    Example:
        >>> from audiomate.corpus import assets
        >>> from audiomate.utils import units import
        >>> ll = assets.LabelList(labels=[
        >>>     assets.Label('music', 0, 2),
        >>>     assets.Label('speech', 2, 5),
        >>>     assets.Label('noise', 4, 6),
        >>>     assets.Label('music', 6, 8)
        >>> ])
        >>> labels = ['speech', 'music', 'noise']
        >>> fs = units.FrameSettings(16000, 16000)
        >>> encoder = FrameOrdinalEncoder(labels, frame_settings=fs)
        >>> encoder.encode(ll)
        array([1,1,0,0,0,2,1,1])
    """

    def __init__(self, labels, frame_settings, sr=None):
        self.labels = labels
        self.frame_settings = frame_settings
        self.sr = sr

    def encode(self, utterance, label_list_idx='default'):
        sr = self.sr or utterance.sampling_rate
        num_samples = utterance.num_samples(sr=sr)
        num_frames = self.frame_settings.num_frames(num_samples)

        mat = np.zeros((num_frames, len(self.labels)))

        if label_list_idx not in utterance.label_lists:
            raise ValueError('Utterance {} has no label-list with idx {}'.format(utterance.idx, label_list_idx))

        label_list = utterance.label_lists[label_list_idx]

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
