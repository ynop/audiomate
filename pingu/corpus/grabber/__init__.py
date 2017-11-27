"""
This module contains classes for easy access/iterate the data in the corpus.

A grabber should implement the methods ``__len__`` and ``__getitem__``, so the data can be access by indexed access or iteration.
"""

from .signal import FramedSignalGrabber
