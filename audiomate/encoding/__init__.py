"""
The encoding module provides functionality to encode labels to use for example for training a DNN.
"""

from .base import Encoder  # noqa: F401

from .frame_based import FrameHotEncoder  # noqa: F401
from .frame_based import FrameOrdinalEncoder  # noqa: F401

from .utterance_based import TokenOrdinalEncoder  # noqa: F401
