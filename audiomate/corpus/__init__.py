"""
This module contains all parts needed for using a corpus. Aside the main corpus class
:py:class:`audiomate.Corpus`, there are different loaders in the :py:mod:`audiomate.corpus.io` and the
assets used in a corpus in :py:mod:`audiomate.corpus.assets`.
"""

from .base import CorpusView  # noqa: F401

from .corpus import Corpus  # noqa: F401

from audiomate.corpus.subset.subview import Subview  # noqa: F401
from audiomate.corpus.subset.subview import MatchingUtteranceIdxFilter  # noqa: F401
