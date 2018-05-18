import collections

import numpy as np

from audiomate.utils import units
from . import label


class Utterance(object):
    """
    An utterance defines a sample of audio. It is part of a file or can span over the whole file.

    Args:
        idx (str): A unique identifier for the utterance within a dataset.
        file (File): The file this utterance is belonging to.
        issuer (Issuer): The issuer this utterance was created from.
        start (float): The start of the utterance within the audio file in seconds. (default 0)
        end (float): The end of the utterance within the audio file in seconds. -1 indicates that
                     the utterance ends at the end of the file. (default -1)
        label_lists (LabelList, list): A single or multiple label-lists.

    Attributes:
        label_lists (dict): A dictionary containing label-lists with the label-list-idx as key.
    """

    __slots__ = ['idx', 'file', 'issuer', 'start', 'end', 'label_lists']

    def __init__(self, idx, file, issuer=None, start=0, end=-1, label_lists=None):
        self.idx = idx
        self.file = file
        self.issuer = issuer
        self.start = start
        self.end = end
        self.label_lists = {}

        if label_lists is not None:
            self.set_label_list(label_lists)

        if self.issuer is not None:
            self.issuer.utterances.add(self)

    @property
    def end_abs(self):
        """
        Return the absolute end of the utterance relative to the signal.
        """
        if self.end == -1:
            return self.file.duration
        else:
            return self.end

    @property
    def duration(self):
        """
        Return the absolute duration in seconds.
        """
        return self.end_abs - self.start

    def num_samples(self, sr=None):
        """
        Return the number of samples.

        Args:
            sr (int): Calculate the number of samples with the given sampling-rate.
                      If None use the native sampling-rate.

        Returns:
            int: Number of samples
        """
        native_sr = self.sampling_rate
        num_samples = units.seconds_to_sample(self.duration, native_sr)

        if sr is not None:
            ratio = float(sr) / native_sr
            num_samples = int(np.ceil(num_samples * ratio))

        return num_samples

    #
    #   Signal
    #

    def read_samples(self, sr=None, offset=0, duration=None):
        """
        Read the samples of the utterance.

        Args:
            sr (int): If None uses the sampling rate given by the file, otherwise resamples to the given sampling rate.
            offset (float): Offset in seconds to read samples from.
            duration (float): If not None read only this number of seconds in maximum.

        Returns:
            np.ndarray: A numpy array containing the samples as a floating point (numpy.float32) time series.
        """

        read_duration = None

        if self.end >= 0:
            read_duration = self.duration

        if offset > 0:
            read_duration -= offset

        if duration is not None:
            read_duration = min(duration, read_duration)

        return self.file.read_samples(sr=sr, offset=self.start + offset, duration=read_duration)

    @property
    def sampling_rate(self):
        """
        Return the sampling rate.
        """
        return self.file.sampling_rate

    #
    #   Labels
    #

    def set_label_list(self, label_lists):
        """
        Set the given label-list for this utterance. If the label-list-idx is not set, ``default`` is used.
        If there is already a label-list with the given idx, it will be overriden.

        Args:
            label_list (LabelList, list): A single or multiple label-lists to add.

        """

        if isinstance(label_lists, label.LabelList):
            label_lists = [label_lists]

        for label_list in label_lists:
            if label_list.idx is None:
                label_list.idx = 'default'

            label_list.utterance = self
            self.label_lists[label_list.idx] = label_list

    def all_label_values(self, label_list_ids=None):
        """
        Return a set of all label-values occurring in this utterance.

        Args:
            label_list_ids (list): If not None, only label-values from label-lists with an id contained in this list
                                   are considered.

        Returns:
             set: A set of distinct label-values.
        """
        values = set()

        for label_list in self.label_lists.values():
            if label_list_ids is None or label_list.idx in label_list_ids:
                values = values.union(label_list.label_values())

        return values

    def label_count(self, label_list_ids=None):
        """
        Return a dictionary containing the number of times, every label-value in this utterance is occurring.

        Args:
            label_list_ids (list): If not None, only labels from label-lists with an id contained in this list
                                   are considered.

        Returns:
            dict: A dictionary containing the number of occurrences with the label-value as key.
        """
        count = collections.defaultdict(int)

        for label_list in self.label_lists.values():
            if label_list_ids is None or label_list.idx in label_list_ids:
                for label_value, label_count in label_list.label_count().items():
                    count[label_value] += label_count

        return count

    def label_total_duration(self, label_list_ids=None):
        """
        Return a dictionary containing the number of seconds, every label-value is occurring in this utterance.

        Args:
            label_list_ids (list): If not None, only labels from label-lists with an id contained in this list
                                   are considered.

        Returns:
            dict: A dictionary containing the number of seconds with the label-value as key.
        """
        duration = collections.defaultdict(float)

        for label_list in self.label_lists.values():
            if label_list_ids is None or label_list.idx in label_list_ids:
                for label_value, label_duration in label_list.label_total_duration().items():
                    duration[label_value] += label_duration

        return duration
