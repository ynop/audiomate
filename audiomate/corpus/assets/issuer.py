import enum


class Gender(enum.Enum):
    UNKNOWN = 'unknown'
    MALE = 'male'
    FEMALE = 'female'


class AgeGroup(enum.Enum):
    UNKNOWN = 'unknown'
    CHILD = 'child'
    YOUTH = 'youth'
    ADULT = 'adult'
    SENIOR = 'senior'


class Issuer(object):
    """
    The issuer represents a person, object or something that produced an utterance.
    Technically the issuer can be used to group utterances which came from the same source.

    Args:
        idx (str): An unique identifier for this issuer within a dataset.
        info (dict): Any additional info for this issuer as dict.

    Attributes:
        utterances (list): List of utterances that this issuer owns.
    """

    def __init__(self, idx, info={}):
        self.idx = idx
        self.info = info
        self.utterances = set()

    def __str__(self):
        return 'Issuer(idx={0}, info={1})'.format(self.idx, self.info)


class Speaker(Issuer):
    """
    The speaker is the person who spoke in a utterance.

    Args:
        idx (str): An unique identifier for this speaker within a dataset.
        info (dict): Any additional info for this speaker as dict.
        age_group (AgeGroup): The age-group of the speaker (child, adult, ...)
        native_language (str): The native language of the speaker. (ISO 639-3)

    Attributes:
        utterances (list): List of utterances that this issuer owns.
    """

    def __init__(self, idx, gender=Gender.UNKNOWN, age_group=AgeGroup.UNKNOWN, native_language=None, info={}):
        super(Speaker, self).__init__(idx, info=info)

        self.gender = gender
        self.age_group = age_group
        self.native_language = native_language

    def __str__(self):
        return 'Speaker(idx={0}, info={1})'.format(self.idx, self.info)


class Artist(Issuer):
    """
    The artist is the person/group who have produced a musical segment in a utterance.

    Args:
        idx (str): An unique identifier for this speaker within a dataset.
        name (str): The name of the artist/band/...
        info (dict): Any additional info for this speaker as dict.

    Attributes:
        utterances (list): List of utterances that this issuer owns.
    """

    def __init__(self, idx, name, info={}):
        super(Artist, self).__init__(idx, info=info)

        self.name = name

    def __str__(self):
        return 'Artist(idx={0}, info={1})'.format(self.idx, self.info)
