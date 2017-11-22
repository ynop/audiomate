class Utterance(object):
    """
    An utterance defines a sample of audio. It is part of a file or can span over the whole file.

    Args:
        idx (str): A unique identifier for the utterance within a dataset.
        file_idx (str): The identifier of the file this utterance is belonging to.
        issuer_idx (str): The identifier of the issuer this utterance was created from.
        start (float): The start of the utterance within the audio file in seconds. (default 0)
        end (float): The end of the utterance within the audio file in seconds. (default -1) (-1 indicates that the utterance ends at the end of the file.
    """

    __slots__ = ['idx', 'file_idx', 'issuer_idx', 'start', 'end']

    def __init__(self, idx, file_idx, issuer_idx=None, start=0, end=-1):
        self.idx = idx
        self.file_idx = file_idx
        self.issuer_idx = issuer_idx
        self.start = start
        self.end = end
