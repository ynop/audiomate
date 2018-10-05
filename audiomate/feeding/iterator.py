import audiomate
from audiomate.corpus import assets


class DataIterator(object):
    """
    An abstract class representing a data-iterator. A data-iterator provides sequential access to data.
    An implementation of a concrete data-iterator should override the methods ``__iter__`` and ``__next__``.

    A sample returned from a data-iterator is a tuple containing the data for this sample from every container.
    The data from different containers is ordered in the way the containers were passed to the DataIterator.

    Args:
        corpus_or_utt_ids (Corpus, list): Either a corpus or a list of utterances.
                                          This defines which utterances are considered for iterating.
        container (list, Container): A single container or a list of containers.
    """

    def __init__(self, corpus_or_utt_ids, containers):
        if isinstance(corpus_or_utt_ids, audiomate.Corpus):
            self.utt_ids = list(corpus_or_utt_ids.utterances.keys())
        else:
            self.utt_ids = corpus_or_utt_ids

        if isinstance(containers, assets.Container):
            self.containers = [containers]
        else:
            self.containers = containers

        if len(self.containers) == 0:
            raise ValueError('At least one container has to be provided!')

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError
