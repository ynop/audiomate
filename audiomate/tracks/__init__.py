"""
This module contains the different implementations of a track.
A track is an abstract representation of an audio signal.

A concrete implementation provides the functionalty
for reading the audio samples from a specific source.
"""

from .track import Track  # noqa: F401
from .file import FileTrack  # noqa: F401
from .container import ContainerTrack  # noqa: F401

from .utterance import Utterance  # noqa: F401
