"""
The assets module contains data-structures that are contained in a corpus.
"""

from .file import File  # noqa: F401
from .utterance import Utterance  # noqa: F401
from .issuer import Issuer  # noqa: F401

from .label import Label  # noqa: F401
from .label import LabelList  # noqa: F401

from .features import FeatureContainer  # noqa: F401
from .features import PartitioningFeatureIterator  # noqa: F401
