import collections
import heapq
import bisect

from .label import Label


class LabelList(object):
    """
    Represents a list of labels which describe an utterance.
    An utterance can have multiple label-lists.

    Args:
        idx (str): An unique identifier for the label-list
                   within a corpus for one utterance.
        labels (list): The list containing the
                       :py:class:`audiomate.annotations.Label`.

    Attributes:
        utterance (Utterance): The utterance this label-list is belonging to.

    Example:
        >>> label_list = LabelList(idx='transcription', labels=[
        >>>     Label('this', 0, 2),
        >>>     Label('is', 2, 4),
        >>>     Label('timmy', 4, 8)
        >>> ])
    """

    __slots__ = ['idx', 'labels', 'utterance']

    def __init__(self, idx='default', labels=[]):
        self.idx = idx
        self.utterance = None

        self.labels = []
        self.extend(labels)

    def append(self, label):
        """
        Add a label to the end of the list.

        Args:
            label (Label): The label to add.
        """
        label.label_list = self
        self.labels.append(label)

    def extend(self, labels):
        """
        Add a list of labels to the end of the list.

        Args:
            labels (list): Labels to add.
        """
        for label in labels:
            self.append(label)

    def ranges(self, yield_ranges_without_labels=False, include_labels=None):
        """
        Generate all ranges of the label-list. A range is defined
        as a part of the label-list for which the same labels are defined.

        Args:
            yield_ranges_without_labels (bool): If True also yields ranges for
                                                which no labels are defined.
            include_labels (list): If not empty, only the label values in
                                   the list will be considered.

        Returns:
            generator: A generator which yields one range
            (tuple start/end/list-of-labels) at a time.

        Example:
            >>> ll = LabelList(labels=[
            >>>     Label('a', 3.2, 4.5),
            >>>     Label('b', 5.1, 8.9),
            >>>     Label('c', 7.2, 10.5),
            >>>     Label('d', 10.5, 14)
            >>> ])
            >>> ranges = ll.ranges()
            >>> next(ranges)
            (3.2, 4.5, [<audiomate.annotations.Label at 0x1090527c8>])
            >>> next(ranges)
            (4.5, 5.1, [])
            >>> next(ranges)
            (5.1, 7.2, [<audiomate.annotations.label.Label at 0x1090484c8>])
        """

        # all label start events
        events = [(l.start, 1, l) for l in self.labels]
        labels_to_end = False
        heapq.heapify(events)

        current_range_labels = []
        current_range_start = -123

        while len(events) > 0:
            next_event = heapq.heappop(events)
            label = next_event[2]

            # Return current range if its not the first event
            # and not the same time as the previous event
            if -1 < current_range_start < next_event[0]:

                if len(current_range_labels) > 0 or yield_ranges_without_labels:
                    yield (
                        current_range_start,
                        next_event[0],
                        list(current_range_labels)
                    )

            # Update labels and add the "end" event
            if next_event[1] == 1:
                if include_labels is None or label.value in include_labels:
                    current_range_labels.append(label)

                    if label.end == -1:
                        labels_to_end = True
                    else:
                        heapq.heappush(events, (label.end, -1, label))
            else:
                current_range_labels.remove(label)

            current_range_start = next_event[0]

        if labels_to_end and len(current_range_labels) > 0:
            yield (current_range_start, -1, list(current_range_labels))

    def label_values(self):
        """
        Return a list of all occuring label values.

        Returns:
            list: Lexicographically sorted list (str) of label values.

        Example:
            >>> ll = LabelList(labels=[
            >>>     Label('a', 3.2, 4.5),
            >>>     Label('b', 5.1, 8.9),
            >>>     Label('c', 7.2, 10.5),
            >>>     Label('d', 10.5, 14),
            >>>     Label('d', 15, 18)
            >>> ])
            >>> ll.label_values()
            ['a', 'b', 'c', 'd']
        """

        all_labels = set([l.value for l in self])
        return sorted(all_labels)

    def label_count(self):
        """
        Return for each label the number of occurrences within the list.

        Returns:
            dict: A dictionary containing for every label-value (key)
            the number of occurrences (value).

        Example:
            >>> ll = LabelList(labels=[
            >>>     Label('a', 3.2, 4.5),
            >>>     Label('b', 5.1, 8.9),
            >>>     Label('a', 7.2, 10.5),
            >>>     Label('b', 10.5, 14),
            >>>     Label('a', 15, 18)
            >>> ])
            >>> ll.label_count()
            {'a': 3 'b': 2}
        """

        occurrences = collections.defaultdict(int)

        for label in self:
            occurrences[label.value] += 1

        return occurrences

    def all_tokens(self, delimiter=' '):
        """
        Return a list of all tokens occurring in the label-list.

        Args:
            delimiter (str): The delimiter used to split labels into tokens
                             (see :meth:`audiomate.annotations.Label.tokenized`).

        Returns:
             :class:`set`: A set of distinct tokens.
        """
        tokens = set()

        for label in self:
            tokens = tokens.union(set(label.tokenized(delimiter=delimiter)))

        return tokens

    def join(self, delimiter=' ', overlap_threshold=0.1):
        """
        Return a string with all labels concatenated together.
        The order of the labels is defined by the start of the label.
        If the overlapping between two labels is greater than ``overlap_threshold``,
        an Exception is thrown.

        Args:
            delimiter (str): A string to join two consecutive labels.
            overlap_threshold (float): Maximum overlap between two consecutive labels.

        Returns:
            str: A string with all labels concatenated together.

        Example:
            >>> ll = LabelList(idx='some', labels=[
            >>>     Label('a', start=0, end=4),
            >>>     Label('b', start=3.95, end=6.0),
            >>>     Label('c', start=7.0, end=10.2),
            >>>     Label('d', start=10.3, end=14.0)
            >>> ])
            >>> ll.join(' - ')
            'a - b - c - d'
        """

        sorted_by_start = sorted(self.labels)
        concat_values = []
        last_label_end = None

        for label in sorted_by_start:
            if last_label_end is None or (last_label_end - label.start < overlap_threshold and last_label_end > 0):
                concat_values.append(label.value)
                last_label_end = label.end
            else:
                raise ValueError('Labels overlap, not able to define the correct order')

        return delimiter.join(concat_values)

    def tokenized(self, delimiter=' ', overlap_threshold=0.1):
        """
        Return a ordered list of tokens based on all labels.
        Joins all token from all labels (``label.tokenized()```).
        If the overlapping between two labels is greater than ``overlap_threshold``,
        an Exception is thrown.

        Args:
            delimiter (str): The delimiter used to split labels into tokens. (default: space)
            overlap_threshold (float): Maximum overlap between two consecutive labels.

        Returns:
            str: A list containing tokens of all labels ordered according to the label order.

        Example:
            >>> ll = LabelList(idx='some', labels=[
            >>>     Label('a d q', start=0, end=4),
            >>>     Label('b', start=3.95, end=6.0),
            >>>     Label('c a', start=7.0, end=10.2),
            >>>     Label('f g', start=10.3, end=14.0)
            >>> ])
            >>> ll.tokenized(delimiter=' ', overlap_threshold=0.1)
            ['a', 'd', 'q', 'b', 'c', 'a', 'f', 'g']
        """

        sorted_by_start = sorted(self.labels)
        tokens = []
        last_label_end = None

        for label in sorted_by_start:
            if last_label_end is None or (last_label_end - label.start < overlap_threshold and last_label_end > 0):
                tokens.extend(label.tokenized(delimiter=delimiter))
                last_label_end = label.end
            else:
                raise ValueError('Labels overlap, not able to define the correct order')

        return tokens

    def label_total_duration(self):
        """
        Return for each distinct label value the total duration of all occurrences.

        Returns:
            dict: A dictionary containing for every label-value (key)
                  the total duration in seconds (value).

        Example:
            >>> ll = LabelList(labels=[
            >>>     Label('a', 3, 5),
            >>>     Label('b', 5, 8),
            >>>     Label('a', 8, 10),
            >>>     Label('b', 10, 14),
            >>>     Label('a', 15, 18.5)
            >>> ])
            >>> ll.label_total_duration()
            {'a': 7.5 'b': 7.0}
        """

        durations = collections.defaultdict(float)

        for label in self:
            durations[label.value] += label.duration

        return durations

    def apply(self, fn):
        """
        Apply the given function `fn` to every label in this label list.
        `fn` is a function of one argument that receives the current label
        which can then be edited in place.

        Args:
            fn (func): Function to apply to every label

        Example:
            >>> ll = LabelList(labels=[
            ...     Label('a_label', 1.0, 2.0),
            ...     Label('another_label', 2.0, 3.0)
            ... ])
            >>> def shift_labels(label):
            ...     label.start += 1.0
            ...     label.end += 1.0
            ...
            >>> ll.apply(shift_labels)
            >>> ll.labels
            [Label(a_label, 2.0, 3.0), Label(another_label, 3.0, 4.0)]
        """
        for label in self.labels:
            fn(label)

    def split(self, cutting_points, shift_times=False):
        """
        Split the label-list into x parts and return them as new label-lists.
        x is defined by the number of cutting-points (``x == len(cutting_points) + 1``)

        The result is a list of label-lists corresponding to each part.
        Label-list 0 contains labels between ``0`` and ``cutting_points[0]``.
        Label-list 1 contains labels between ``cutting_points[0]`` and ``cutting_points[1]``.
        And so on.

        Args:
            cutting_points (list): List of floats defining the points in seconds,
                                   where the label-list is splitted.
            shift_times (bool): If True, start and end-time of shifted in splitted label-lists.
                                 So the start is relative to the cutting point and
                                 not to the beginning of the original label-list.

        Returns:
            list: A list of of :class:`audiomate.annotations.LabelList`.

        Example:

            >>> ll = LabelList(labels=[
            >>>     Label('a', 0, 5),
            >>>     Label('b', 5, 10),
            >>>     Label('c', 11, 15),
            >>> ])
            >>>
            >>> res = ll.split([4.1, 8.9, 12.0])
            >>> len(res)
            4
            >>> res[0].labels
            [Label('a', 0.0, 4.1)]
            >>> res[1].labels
            [
                Label('a', 4.1, 5.0),
                Label('b', 5.0, 8.9)
            ]
            >>> res[2].labels
            [
                Label('b', 8.9, 10.0),
                Label('c', 11.0, 12.0)
            ]
            >>> res[3].labels
            [Label('c', 12.0, 15.0)]

        If ``shift_times=True``, the times are adjusted to be relative
        to the cutting-points for every label-list but the first.

            >>> ll = LabelList(labels=[
            >>>     Label('a', 0, 5),
            >>>     Label('b', 5, 10),
            >>> ])
            >>>
            >>> res = ll.split([4.6])
            >>> len(res)
            4
            >>> res[0].labels
            [Label('a', 0.0, 4.6)]
            >>> res[1].labels
            [
                Label('a', 0.0, 0.4),
                Label('b', 0.4, 5.4)
            ]
        """

        if len(cutting_points) == 0:
            raise ValueError('At least one cutting-point is needed!')

        cutting_points = sorted(cutting_points)

        label_lists = [LabelList(idx=self.idx) for _ in range(len(cutting_points) + 1)]

        for label in sorted(self.labels):
            if label.end < 0:
                # if label end is unknown (end of utt) we assume its past the last cutting point
                label_end = cutting_points[-1] + 1000.0
            else:
                label_end = label.end

            # find indices where start and end of label would be inserted in cutting_points
            start_cut_index = bisect.bisect_right(cutting_points, label.start)
            end_cut_index = bisect.bisect_left(cutting_points, label_end)

            if end_cut_index <= start_cut_index:
                # label is between two cutting points so append to label-list with that index
                new_label = Label(label.value, start=label.start, end=label.end)

                if shift_times and start_cut_index > 0:
                    new_label.start -= cutting_points[start_cut_index - 1]

                    if new_label.end > 0:
                        new_label.end -= cutting_points[start_cut_index - 1]

                label_lists[start_cut_index].append(new_label)
            else:
                # Cutting-points with index between start_cut_index, end_cut_index are within current label
                # Therefore we split the label
                for index in range(start_cut_index, end_cut_index + 1):
                    if index == start_cut_index:
                        sub_label_start = label.start
                    else:
                        sub_label_start = cutting_points[index - 1]

                    if index >= end_cut_index:
                        sub_label_end = label.end
                    else:
                        sub_label_end = cutting_points[index]

                    if shift_times and index > 0:
                        sub_label_start -= cutting_points[index - 1]

                        if sub_label_end > 0:
                            sub_label_end -= cutting_points[index - 1]

                    new_label = Label(label.value, start=sub_label_start, end=sub_label_end)
                    label_lists[index].append(new_label)

        return label_lists

    def __getitem__(self, item):
        return self.labels.__getitem__(item)

    def __iter__(self):
        return self.labels.__iter__()

    def __len__(self):
        return self.labels.__len__()

    @classmethod
    def create_single(cls, value, idx='default'):
        """ Create a label-list with a single label containing the given value. """

        return LabelList(idx=idx, labels=[
            Label(value=value)
        ])
