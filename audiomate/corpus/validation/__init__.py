"""
This module contains functions for validating a corpus on different properties.
e.g. if the length of the utterance is to short for its corresponding transcription.

:py:class:`audiomate.corpus.validation.Validator` is the base class for performing validations.
It can be extended to implement validators for specific tests/validations.
Thre result of every validator has to be a :py:class:`audiomate.corpus.validation.ValidationResult`
or a subclass of it.
"""

from .base import Validator  # noqa: F401
from .base import ValidationResult  # noqa: F401
from .base import InvalidUtterancesResult  # noqa: F401

from .combine import CombinedValidator  # noqa: F401
from .combine import CombinedValidationResult  # noqa: F401

from .label_list import UtteranceTranscriptionRatioValidator  # noqa: F401
from .label_list import LabelCountValidator  # noqa: F401
from .label_list import LabelCoverageValidator  # noqa: F401
from .label_list import LabelCoverageValidationResult  # noqa: F401
from .label_list import LabelOverflowValidator  # noqa: F401
from .label_list import LabelOverflowValidationResult  # noqa: F401
