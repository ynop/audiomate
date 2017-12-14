import collections
import heapq


class Label(object):
    """
    Represents a label that describes some part of an utterance.

    Parameters:
        value (str): The text of the label.
        start (float): Start of the label within the utterance in seconds. (default: 0)
        end (float): End of the label within the utterance in seconds. (default: -1) (-1 defines
                     the end of the utterance)
    """
    __slots__ = ['value', 'start', 'end']

    def __init__(self, value, start=0, end=-1):
        self.value = value
        self.start = start
        self.end = end


class LabelList(object):
    """
    Represents a list of labels which describe an utterance.
    An utterance can have multiple label-lists.

    Args:
        idx (str): An unique identifier for the label-list within a corpus for one utterance.

    Attributes:
        labels (list): The list containing the :py:class:`pingu.corpus.assets.Label`.

    Example::

        >>> label_list = LabelList(idx='transcription', labels=[
        >>>     Label('this', 0, 2),
        >>>     Label('is', 2, 4),
        >>>     Label('timmy', 4, 8)
        >>> ])
    """

    __slots__ = ['labels', 'idx']

    def __init__(self, idx='default', labels=[]):
        self.idx = idx
        self.labels = list(labels)

    def append(self, label):
        """
        Add a label to the end of the list.

        Args:
            label (Label): The label to add.
        """
        self.labels.append(label)

    def extend(self, labels):
        """
        Add a list of labels to the end of the list.

        Args:
            labels (list): Labels to add.
        """
        self.labels.extend(labels)

    def ranges(self, yield_ranges_without_labels=False, include_labels=None):
        """
        Generate all ranges of the label-list. A range is defined as a part of the label-list for
        which the same labels are defined.

        Args:
            yield_ranges_without_labels (bool): If True also yields ranges for which no labels are
                                                defined.
            include_labels (list): If not empty, only the label values in the list will be
                                   considered.

        Returns:
            generator: A generator which yields one range (tuple start/end/list-of-labels) at a
                       time.

        Example:
            >>> ll = assets.LabelList(labels=[
            >>>     assets.Label('a', 3.2, 4.5),
            >>>     assets.Label('b', 5.1, 8.9),
            >>>     assets.Label('c', 7.2, 10.5),
            >>>     assets.Label('d', 10.5, 14)
            >>> ])
            >>> ranges = ll.ranges()
            >>> next(ranges)
            (3.2, 4.5, [<pingu.corpus.assets.label.Label at 0x1090527c8>])
            >>> next(ranges)
            (4.5, 5.1, [])
            >>> next(ranges)
            (5.1, 7.2, [<pingu.corpus.assets.label.Label at 0x1090484c8>])
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

            # Return current range if its not the first event and not the same time as the previous
            # event
            if -1 < current_range_start < next_event[0]:

                if len(current_range_labels) > 0 or yield_ranges_without_labels:
                    yield (current_range_start, next_event[0], list(current_range_labels))

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
            >>> ll = assets.LabelList(labels=[
            >>>     assets.Label('a', 3.2, 4.5),
            >>>     assets.Label('b', 5.1, 8.9),
            >>>     assets.Label('c', 7.2, 10.5),
            >>>     assets.Label('d', 10.5, 14),
            >>>     assets.Label('d', 15, 18)
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
            dict: A dictionary container for every label-value (key) the number of occurrences
                  (value).

        Example::
            >>> ll = assets.LabelList(labels=[
            >>>     assets.Label('a', 3.2, 4.5),
            >>>     assets.Label('b', 5.1, 8.9),
            >>>     assets.Label('a', 7.2, 10.5),
            >>>     assets.Label('b', 10.5, 14),
            >>>     assets.Label('a', 15, 18)
            >>> ])
            >>> ll.label_count()
            {'a': 3 'b': 2}
        """

        occurrences = collections.defaultdict(int)

        for label in self:
            occurrences[label.value] += 1

        return occurrences

    def __getitem__(self, item):
        return self.labels.__getitem__(item)

    def __iter__(self):
        return self.labels.__iter__()

    def __len__(self):
        return self.labels.__len__()
