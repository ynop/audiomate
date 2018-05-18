"""
This module contains base classes for offline processing pipeline steps.
"""

import abc

from audiomate.corpus.preprocessing import processor
from audiomate.corpus.preprocessing.pipeline import base


class OfflineComputation(base.Computation, processor.OfflineProcessor, metaclass=abc.ABCMeta):
    """
    Base class for a computation step in a offline processing pipeline.
    """

    def __init__(self, parent=None, name=None):
        if parent is not None and not (isinstance(parent, OfflineComputation) or isinstance(parent, OfflineReduction)):
            raise ValueError('Cannot combine offline step with other steps.')
        super(OfflineComputation, self).__init__(parent=parent, name=name)

    def process_sequence(self, frames, sampling_rate, utterance=None, corpus=None):
        return self.process(frames, sampling_rate, utterance=utterance, corpus=corpus)

    def process(self, frames, sampling_rate, corpus=None, utterance=None):
        return super(OfflineComputation, self).process(frames, sampling_rate, corpus=corpus, utterance=utterance)

    @abc.abstractmethod
    def compute(self, frames, sampling_rate, corpus=None, utterance=None):
        pass


class OfflineReduction(base.Reduction, processor.OfflineProcessor, metaclass=abc.ABCMeta):
    """
    Base class for a reduction step in a offline processing pipeline.
    """

    def __init__(self, parents, name=None):
        for parent in parents:
            if not (isinstance(parent, OfflineComputation) or isinstance(parent, OfflineReduction)):
                raise ValueError('Cannot combine offline step with other steps.')

        super(OfflineReduction, self).__init__(parents, name=name)

    def process_sequence(self, frames, sampling_rate, utterance=None, corpus=None):
        return self.process(frames, sampling_rate, utterance=utterance, corpus=corpus)

    def process(self, frames, sampling_rate, corpus=None, utterance=None):
        return super(OfflineReduction, self).process(frames, sampling_rate, corpus=corpus, utterance=utterance)

    @abc.abstractmethod
    def compute(self, frames, sampling_rate, corpus=None, utterance=None):
        pass
