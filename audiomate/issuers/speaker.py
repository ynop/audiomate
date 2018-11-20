import enum

from .issuer import Issuer


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

    __slots__ = ['gender', 'age_group', 'native_language']

    def __init__(self, idx, gender=Gender.UNKNOWN, age_group=AgeGroup.UNKNOWN,
                 native_language=None, info={}):
        super(Speaker, self).__init__(idx, info=info)

        self.gender = gender
        self.age_group = age_group
        self.native_language = native_language

    def __str__(self):
        return 'Speaker(idx={0}, info={1})'.format(self.idx, self.info)
