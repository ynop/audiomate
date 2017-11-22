class Label(object):
    """
    Represents a label that describes some part of an utterance.

    Parameters:
        value (str): The text of the label.
        start (float): Start of the label within the utterance in seconds. (default: 0)
        end (float): End of the label within the utterance in seconds. (default: -1) (-1 defines the end of the utterance)
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

    def __getitem__(self, item):
        return self.labels.__getitem__(item)

    def __iter__(self):
        return self.labels.__iter__()

    def __len__(self):
        return self.labels.__len__()
