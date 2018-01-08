import abc


class Grabber(metaclass=abc.ABCMeta):
    """
    A grabber is used for accessing the corpus data via indexing.
    The grabber defines the ``__len__`` method, which returns the number of samples and the ``__getitem__`` method,
    which returns a sample for the given index.
    """

    def __init__(self, corpus, feature_container):
        self.corpus = corpus

        if isinstance(feature_container, str):
            self.feature_container = self.corpus.feature_containers[feature_container]
        else:
            self.feature_container = feature_container

    @abc.abstractmethod
    def __len__(self):
        return 0

    @abc.abstractmethod
    def __getitem__(self, item):
        return None
