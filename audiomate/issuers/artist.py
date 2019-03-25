import copy

from .issuer import Issuer


class Artist(Issuer):
    """
    The artist is the person/group who have produced a musical segment
    in a utterance.

    Args:
        idx (str): An unique identifier for this speaker within a dataset.
        name (str): The name of the artist/band/...
        info (dict): Any additional info for this speaker as dict.

    Attributes:
        Issuer.utterances (list): List of utterances that this issuer owns.
    """

    __slots__ = ['name']

    def __init__(self, idx, name, info={}):
        super(Artist, self).__init__(idx, info=info)

        self.name = name

    def __str__(self):
        return 'Artist(idx={0}, info={1})'.format(self.idx, self.info)

    def __copy__(self):
        # self.utterances is ignored intentionally
        # only a "weak-ref" when added to a corpus

        cp = Artist(
            self.idx,
            self.name,
            info=self.info
        )

        return cp

    def __deepcopy__(self, memo):
        # self.utterances is ignored intentionally
        # only a "weak-ref" when added to a corpus

        cp = Artist(
            self.idx,
            self.name,
            info=copy.deepcopy(self.info, memo)
        )

        return cp
