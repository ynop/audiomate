"""
This module contains all parts needed for using a corpus. Aside the main corpus class
:py:class:`audiomate.Corpus`, there are different loaders in the :py:mod:`audiomate.corpus.io`.
"""

from .base import CorpusView  # noqa: F401

from .corpus import Corpus  # noqa: F401

from audiomate.corpus.subset.subview import Subview  # noqa: F401
from audiomate.corpus.subset.subview import MatchingUtteranceIdxFilter  # noqa: F401

# Definition of common Label-List identifiers
LL_DOMAIN = 'domain'
LL_DOMAIN_MUSIC = 'music'
LL_DOMAIN_SPEECH = 'speech'
LL_DOMAIN_NOISE = 'noise'

LL_WORD_TRANSCRIPT = 'word-transcript'
LL_WORD_TRANSCRIPT_RAW = 'word-transcript-raw'
LL_WORD_TRANSCRIPT_ALIGNED = 'word-transcript-aligned'

LL_PHONE_TRANSCRIPT = 'phone-transcript'
LL_PHONE_TRANSCRIPT_ALIGNED = 'phone-transcript-aligned'

LL_GENRE = 'genre'

LL_SOUND_CLASS = 'sound-class'
