"""
This module contains classes for easy access/iterate the data in the corpus.

The :py:class:`pingu.corpus.grabber.Grabber` is the base for classes, which provide indexed access to the corpus data.
"""

from .base import Grabber  # noqa: 401

from .signal import FramedSignalGrabber  # noqa: 401
from .frames import FrameClassificationGrabber  # noqa: 401
