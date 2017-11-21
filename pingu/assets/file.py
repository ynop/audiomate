class File(object):
    __slots__ = ['idx', 'path']

    def __init__(self, idx, path):
        self.idx = idx
        self.path = path
