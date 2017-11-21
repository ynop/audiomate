class Issuer(object):
    """
    The issuer represents a person, object or something that produced an utterance.
    Technically the issuer can be used to group utterances which came from the same source.

    Arguments:
        idx: An unique identifier for this issuer within a dataset.
        info: Any additional infos for this issuer as dict.
    """

    def __init__(self, idx, info={}):
        self.idx = idx
        self.info = info
