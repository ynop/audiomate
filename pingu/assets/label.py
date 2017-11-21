class Label(object):
    """
    Represents a label that describes some part of an utterance.

    Parameters:
        value: The text of the label.
        start: Start of the label within the utterance in seconds. (Optional)
        end: End of the label within the utterance in seconds. (Optional)
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

    Parameters:
        idx: An unique identifier for the label-list within an utterance.

    Attributes:
        labels: The list containing the labels.
    """

    __slots__ = ['labels', 'idx']

    def __init__(self, idx='default'):
        self.idx = idx
        self.labels = []

    def append(self, label):
        self.labels.append(label)

    def extend(self, labels):
        self.labels.extend(labels)

    def __getitem__(self, item):
        return self.labels.__getitem__(item)

    def __iter__(self):
        return self.labels.__iter__()

    def __len__(self):
        return self.labels.__len__()
