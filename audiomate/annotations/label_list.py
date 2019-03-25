import collections
import copy

import intervaltree

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
        label_tree (IntervalTree): The interval-tree storing the labels.

    Example:
        >>> label_list = LabelList(idx='transcription', labels=[
        >>>     Label('this', 0, 2),
        >>>     Label('is', 2, 4),
        >>>     Label('timmy', 4, 8)
        >>> ])
    """

    __slots__ = ['idx', 'label_tree', 'utterance']

    def __init__(self, idx='default', labels=None):
        self.idx = idx
        self.utterance = None

        self.label_tree = intervaltree.IntervalTree()

        if labels is not None:
            self.update(labels)

    def __eq__(self, other):
        data_this = (self.idx, self.label_tree)
        data_other = (other.idx, other.label_tree)
        return data_this == data_other

    def __iter__(self):
        for interval in self.label_tree:
            yield interval.data

    def __len__(self):
        return self.label_tree.__len__()

    def __copy__(self):
        # utterance is ignored intentionally,
        # since it is kind of a weak ref
        return LabelList(
            idx=self.idx,
            labels=[iv.data for iv in self.label_tree]
        )

    def __deepcopy__(self, memo):
        # utterance is ignored intentionally,
        # since it is kind of a weak ref
        return LabelList(
            idx=self.idx,
            labels=copy.deepcopy([iv.data for iv in self.label_tree], memo)
        )

    @property
    def labels(self):
        """ Return list of labels. """
        return list(self)

    @property
    def start(self):
        """ Return start of the earliest starting label (lower bound). """
        return self.label_tree.begin()

    @property
    def end(self):
        """ Return end of the lastly ending label (upper bound). """
        return self.label_tree.end()

    #
    #   Alteration
    #

    def add(self, label):
        """
        Add a label to the end of the list.

        Args:
            label (Label): The label to add.
        """
        label.label_list = self
        self.label_tree.addi(label.start, label.end, label)

    def addl(self, value, start=0.0, end=float('inf')):
        """
        Shortcut for ``add(Label(value, start, end))``.
        """
        self.add(Label(value, start=start, end=end))

    def update(self, labels):
        """
        Add a list of labels to the end of the list.

        Args:
            labels (list): Labels to add.
        """
        for label in labels:
            self.add(label)

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

    def merge_overlaps(self, threshold=0.0):
        """
        Merge overlapping labels with the same value.
        Two labels are considered overlapping,
        if ``l2.start - l1.end < threshold``.

        Args:
            threshold (float): Maximal distance between two labels
                               to be considered as overlapping.
                               (default: 0.0)

        Example:
            >>> ll = LabelList(labels=[
            ...     Label('a_label', 1.0, 2.0),
            ...     Label('a_label', 1.5, 2.7),
            ...     Label('b_label', 1.0, 2.0),
            ... ])
            >>> ll.merge_overlapping_labels()
            >>> ll.labels
            [
                Label('a_label', 1.0, 2.7),
                Label('b_label', 1.0, 2.0),
            ]
        """

        updated_labels = []
        all_intervals = self.label_tree.copy()

        # recursivly find a group of overlapping labels with the same value
        def recursive_overlaps(interval):
            range_start = interval.begin - threshold
            range_end = interval.end + threshold

            direct_overlaps = all_intervals.overlap(range_start, range_end)
            all_overlaps = [interval]
            all_intervals.discard(interval)

            for overlap in direct_overlaps:
                if overlap.data.value == interval.data.value:
                    all_overlaps.extend(recursive_overlaps(overlap))

            return all_overlaps

        # For every remaining interval
        # - Find overlapping intervals recursively
        # - Remove them
        # - Create a concatenated new label
        while not all_intervals.is_empty():
            next_interval = list(all_intervals)[0]
            overlapping = recursive_overlaps(next_interval)

            ov_start = float('inf')
            ov_end = 0.0
            ov_value = next_interval.data.value

            for overlap in overlapping:
                ov_start = min(ov_start, overlap.begin)
                ov_end = max(ov_end, overlap.end)
                all_intervals.discard(overlap)

            updated_labels.append(Label(
                ov_value,
                ov_start,
                ov_end
            ))

        # Replace the old labels with the updated ones
        self.label_tree.clear()
        self.update(updated_labels)

    #
    #   Statistics
    #

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

    #
    #   Query Label Values
    #

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

    #
    #   Restructuring
    #

    def separated(self):
        """
        Create a separate Label-List for every distinct label-value.

        Returns:
            dict: A dictionary with distinct label-values as keys.
                  Every value is a LabelList containing only labels with the same value.

        Example:
            >>> ll = LabelList(idx='some', labels=[
            >>>     Label('a', start=0, end=4),
            >>>     Label('b', start=3.95, end=6.0),
            >>>     Label('a', start=7.0, end=10.2),
            >>>     Label('b', start=10.3, end=14.0)
            >>> ])
            >>> s = ll.separate()
            >>> s['a'].labels
            [Label('a', start=0, end=4), Label('a', start=7.0, end=10.2)]
            >>> s['b'].labels
            [Label('b', start=3.95, end=6.0), Label('b', start=10.3, end=14.0)]
        """
        separated_lls = collections.defaultdict(LabelList)

        for label in self.labels:
            separated_lls[label.value].add(label)

        for ll in separated_lls.values():
            ll.idx = self.idx

        return separated_lls

    def labels_in_range(self, start, end, fully_included=False):
        """
        Return a list of labels, that are within the given range.
        Also labels that only overlap are included.

        Args:
            start(float): Start-time in seconds.
            end(float): End-time in seconds.
            fully_included(bool): If ``True``, only labels fully included
                                   in the range are returned. Otherwise
                                   also overlapping ones are returned.
                                   (default ``False``)

        Returns:
            list: List of labels in the range.

        Example:
            >>> ll = LabelList(labels=[
            >>>     Label('a', 3.2, 4.5),
            >>>     Label('b', 5.1, 8.9),
            >>>     Label('c', 7.2, 10.5),
            >>>     Label('d', 10.5, 14)
            >>>])
            >>> ll.labels_in_range(6.2, 10.1)
            [Label('b', 5.1, 8.9), Label('c', 7.2, 10.5)]
        """

        if fully_included:
            intervals = self.label_tree.envelop(start, end)
        else:
            intervals = self.label_tree.overlap(start, end)

        return [iv.data for iv in intervals]

    def ranges(self, yield_ranges_without_labels=False, include_labels=None):
        """
        Generate all ranges of the label-list. A range is defined
        as a part of the label-list for which the same labels are defined.

        Args:
            yield_ranges_without_labels(bool): If True also yields ranges for
                                                which no labels are defined.
            include_labels(list): If not empty, only the label values in
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
            >>>])
            >>> ranges = ll.ranges()
            >>> next(ranges)
            (3.2, 4.5, [ < audiomate.annotations.Label at 0x1090527c8 > ])
            >>> next(ranges)
            (4.5, 5.1, [])
            >>> next(ranges)
            (5.1, 7.2, [ < audiomate.annotations.label.Label at 0x1090484c8 > ])
        """

        tree_copy = self.label_tree.copy()

        # Remove labels not included
        if include_labels is not None:
            for iv in list(tree_copy):
                if iv.data.value not in include_labels:
                    tree_copy.remove(iv)

        def reduce(x, y):
            x.append(y)
            return x

        # Split labels when overlapping and merge equal ranges to a list of labels
        tree_copy.split_overlaps()
        tree_copy.merge_equals(data_reducer=reduce, data_initializer=[])

        intervals = sorted(tree_copy)
        last_end = intervals[0].begin

        # yield range by range
        for i in range(len(intervals)):
            iv = intervals[i]

            # yield an empty range if necessary
            if yield_ranges_without_labels and iv.begin > last_end:
                yield (last_end, iv.begin, [])

            yield (iv.begin, iv.end, iv.data)

            last_end = iv.end

    def split(self, cutting_points, shift_times=False, overlap=0.0):
        """
        Split the label-list into x parts and return them as new label-lists.
        x is defined by the number of cutting-points(``x == len(cutting_points) + 1``)

        The result is a list of label-lists corresponding to each part.
        Label-list 0 contains labels between ``0`` and ``cutting_points[0]``.
        Label-list 1 contains labels between ``cutting_points[0]`` and ``cutting_points[1]``.
        And so on.

        Args:
            cutting_points(list): List of floats defining the points in seconds,
                                  where the label-list is splitted.
            shift_times(bool): If True, start and end-time are shifted in splitted label-lists.
                               So the start is relative to the cutting point and
                               not to the beginning of the original label-list.
            overlap(float): Amount of overlap in seconds. This amount is subtracted
                            from a start-cutting-point, and added to a end-cutting-point.

        Returns:
            list: A list of of: class: `audiomate.annotations.LabelList`.

        Example:

            >>> ll = LabelList(labels=[
            >>>     Label('a', 0, 5),
            >>>     Label('b', 5, 10),
            >>>     Label('c', 11, 15),
            >>>])
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

        If ``shift_times = True``, the times are adjusted to be relative
        to the cutting-points for every label-list but the first.

            >>> ll = LabelList(labels=[
            >>>     Label('a', 0, 5),
            >>>     Label('b', 5, 10),
            >>>])
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

        # we have to loop in sorted order
        cutting_points = sorted(cutting_points)

        splits = []
        iv_start = 0.0

        for i in range(len(cutting_points) + 1):
            if i < len(cutting_points):
                iv_end = cutting_points[i]
            else:
                iv_end = float('inf')

            # get all intervals intersecting range
            intervals = self.label_tree.overlap(
                iv_start - overlap,
                iv_end + overlap
            )

            cp_splits = LabelList(idx=self.idx)

            # Extract labels from intervals with updated times
            for iv in intervals:
                label = copy.deepcopy(iv.data)
                label.start = max(0, iv_start - overlap, label.start)
                label.end = min(iv_end + overlap, label.end)

                if shift_times:
                    orig_start = max(0, iv_start - overlap)
                    label.start -= orig_start
                    label.end -= orig_start

                cp_splits.add(label)

            splits.append(cp_splits)
            iv_start = iv_end

        return splits

    #
    #   Convenience Constructors
    #

    @classmethod
    def create_single(cls, value, idx='default'):
        """ Create a label-list with a single label containing the given value. """

        return LabelList(idx=idx, labels=[
            Label(value=value)
        ])

    @classmethod
    def with_label_values(cls, values, idx='default'):
        """
        Create a new label-list containing labels with the given values.
        All labels will have default start/end values of 0 and ``inf``.

        Args:
            values(list): List of values(str) that should be created and appended
                           to the label-list.
            idx(str): The idx of the label-list.

        Returns:
            (LabelList): New label-list.

        Example:
            >>> ll = LabelList.with_label_values(['a', 'x', 'z'], idx='letters')
            >>> ll.idx
            'letters'
            >>> ll.labels
            [
                Label('a', 0, inf),
                Label('x', 0, inf),
                Label('z', 0, inf),
            ]
        """

        ll = LabelList(idx=idx)

        for label_value in values:
            ll.add(Label(label_value))

        return ll
