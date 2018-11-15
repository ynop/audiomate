"""
This module contains the different implementations of containers.
A container is normally used to store data of a specific type
for all instances of a corpus (e.g. mfcc-features of all utterances).

All container implementations are based on
:py:class:`audiomate.containers.Container`, which provides the basic
functionality to access a HDF5-file using h5py.
"""

from .container import Container  # noqa: F401
from .features import FeatureContainer  # noqa: F401
from .audio import AudioContainer  # noqa: F401
