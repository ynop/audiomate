class File(object):
    """
    The file object is used to hold any data/infos about a file contained in a corpus.

    Args:
        idx (str): A unique identifier within a corpus for the file.
        path (str): The path to the file.
    """
    __slots__ = ['idx', 'path']

    def __init__(self, idx, path):
        self.idx = idx
        self.path = path
