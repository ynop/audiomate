import collections


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

    Attributes:
        label_lists (dict): A dictionary containing label-lists with the label-list-idx as key.
    """

    __slots__ = ['idx', 'file', 'issuer', 'start', 'end', 'label_lists']

    def __init__(self, idx, file, issuer=None, start=0, end=-1):
        self.idx = idx
        self.file = file
        self.issuer = issuer
        self.start = start
        self.end = end
        self.label_lists = {}

        if self.issuer is not None:
            self.issuer.utterances.add(self)

    #
    #   Signal
    #

    def read_samples(self, sr=None):
        """
        Read the samples of the utterance.

        Args:
            sr (int): If None uses the sampling rate given by the file, otherwise resamples to the given sampling rate.

        Returns:
            np.ndarray: A numpy array containing the samples as a floating point (numpy.float32) time series.
        """
        duration = None

        if self.end >= 0:
            duration = self.end - self.start

        return self.file.read_samples(sr=sr, offset=self.start, duration=duration)

    #
    #   Labels
    #

    def set_label_list(self, label_list):
        """
        Set the given label-list for this utterance. If the label-list-idx is not set, ``default`` is used.
        If there is already a label-list with the given idx, it will be overriden.

        Args:
            label_list (LabelList): The label-list to add.

        """
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
                for label_value, label_list_count in label_list.label_count().items():
                    count[label_value] += label_list_count

        return count
