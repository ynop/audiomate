"""
This module contains all parts needed for using a corpus. Aside the main corpus class
:py:class:`pingu.Corpus`, there are different loaders in the :py:mod:`pingu.corpus.io` and the
assets used in a corpus in :py:mod:`pingu.corpus.assets`.
"""

from .base import CorpusView  # noqa: F401

from .corpus import Corpus  # noqa: F401

from pingu.corpus.subset.subview import Subview  # noqa: F401
from pingu.corpus.subset.subview import MatchingUtteranceIdxFilter  # noqa: F401
