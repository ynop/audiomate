"""
The :mod:`audiomate.feeding` module provides tools for a simple access to data stored in different
:class:`audiomate.corpus.assets.Container`.
"""

from .dataset import Dataset  # noqa: F401
from .dataset import UtteranceDataset  # noqa: F401
from .dataset import FrameDataset  # noqa: F401
from .dataset import MultiFrameDataset  # noqa: F401

from .iterator import DataIterator  # noqa: F401
from .iterator import FrameIterator  # noqa: F401
from .iterator import MultiFrameIterator  # noqa: F401

from .partitioning import PartitioningFeatureIterator  # noqa: F401

from .partitioning import PartitioningContainerLoader  # noqa: F401
from .partitioning import PartitionInfo  # noqa: F401
from .partitioning import PartitionData  # noqa: F401
