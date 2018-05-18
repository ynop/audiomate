"""
This module provides building blocks for preprocessing blocks/pipelines.
A pipeline is built out of processors, which process the samples of an utterance.

There are different levels of abstractions for a processor.
On top there is the :py:class:`audiomate.corpus.preprocessing.Processor`, which provides the basic structure to process
the utterances one after another.

Based on that there are two subclasses :py:class:`audiomate.corpus.preprocessing.OnlineProcessor` and
:py:class:`audiomate.corpus.preprocessing.OfflineProcessor`. The first one is streaming-processor for processing samples
frame by frame without the need for loading the full utterance.

The latter one is used to process all samples of the utterance at once.
In a lot of cases this is easier to implement.
"""

from .processor import Processor  # noqa: F401
from .processor import OfflineProcessor  # noqa: F401
