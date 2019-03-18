import copy


class Issuer(object):
    """
    The issuer represents a person, object or something that produced an
    utterance. Technically the issuer can be used to group utterances
    that came from the same source.

    Args:
        idx (str): An unique identifier for this issuer within a dataset.
        info (dict): Any additional info for this issuer as dict.

    Attributes:
        Issuer.utterances (list): List of utterances that this issuer owns.
    """

    __slots__ = ['idx', 'info', 'utterances']

    def __init__(self, idx, info={}):
        self.idx = idx
        self.info = info
        self.utterances = set()

    def __str__(self):
        return 'Issuer(idx={0}, info={1})'.format(self.idx, self.info)

    def __copy__(self):
        # self.utterances is ignored intentionally
        # only a "weak-ref" when added to a corpus

        cp = Issuer(
            self.idx,
            info=self.info
        )

        return cp

    def __deepcopy__(self, memo):
        # self.utterances is ignored intentionally
        # only a "weak-ref" when added to a corpus

        cp = Issuer(
            self.idx,
            info=copy.deepcopy(self.info, memo)
        )

        return cp
