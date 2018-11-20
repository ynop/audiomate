"""
This module contains classes for creating frame processing pipelines.

A pipeline consists of one of two types of steps. A computation step takes data from a previous step or the input and
processes it. A reduction step is used to merge outputs of multiple previous steps.
It takes outputs of all incoming steps and outputs a single data block.

The steps are managed as a directed graph,
which is built by passing the parent steps to the ``__init__`` method of a step.
Every step that is created has his own graph, but inherits all nodes and edges of the graphs of his parent steps.

Every pipeline represents a processor and implements the ``process_frames`` method.
"""

from .base import Chunk  # noqa: F401
from .base import Step  # noqa: F401
from .base import Computation  # noqa: F401
from .base import Reduction  # noqa: F401

from .normalization import MeanVarianceNorm  # noqa: F401

from .spectral import MelSpectrogram  # noqa: F401
from .spectral import MFCC  # noqa: F401

from .magnitude_scaling import PowerToDb  # noqa: F401

from .varia import Delta  # noqa: F401
from .varia import AddContext  # noqa: F401

from .reduction import Stack  # noqa: F401

from .pool import AvgPool  # noqa: F401
from .pool import VarPool  # noqa: F401

from .onset import OnsetStrength  # noqa: F401
from .rhythm import Tempogram  # noqa: F401
