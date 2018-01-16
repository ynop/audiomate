"""
This module provides class for handling subviews.
This includes the subview class itself and also the FilterCriterion classes, which are used to
define the data contained in a subview.
"""
import abc

from . import base


class FilterCriterion(metaclass=abc.ABCMeta):
    """
    A filter criterion decides wheter a given utterance contained in a given corpus matches the
    filter.
    """

    def match(self, utterance, corpus):
        """
        Check if the utterance matches the filter.

        Args:
            utterance (Utterance): The utterance to match.
            corpus (CorpusView): The corpus that contains the utterance.

        Returns:
            bool: True if the filter matches the utterance, False otherwise.
        """
        pass

    @abc.abstractmethod
    def serialize(self):
        """
        Serialize this filter criterion to write to a file.
        The output needs to be a single line without line breaks.

        Returns:
            str: A string representing this filter criterion.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def parse(cls, representation):
        """
        Create a filter criterion based on a string representation (created with ``serialize``).

        Args:
            representation (str): The string representation.

        Returns:
            FilterCriterion: The filter criterion from that representation.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def name(cls):
        """
        Returns a name identifying this type of filter criterion.
        """
        return 'unknown'


class MatchingUtteranceIdxFilter(FilterCriterion):
    """
    A filter criterion that matches utterances based on utterance-ids.

    Args:
        utterance_idxs (set): A list of utterance-ids. Only utterances in the list will pass the
                               filter
        inverse (bool): If True only utterance not in the list pass the filter.
    """

    def __init__(self, utterance_idxs=set(), inverse=False):
        self.utterance_idxs = utterance_idxs
        self.inverse = inverse

    def match(self, utterance, corpus):
        return (utterance.idx in self.utterance_idxs and not self.inverse) \
               or (utterance.idx not in self.utterance_idxs and self.inverse)

    def serialize(self):
        inverse_indication = 'exclude' if self.inverse else 'include'
        id_string = ','.join(sorted(self.utterance_idxs))
        return '{},{}'.format(inverse_indication, id_string)

    @classmethod
    def parse(cls, representation):
        items = representation.strip().split(',')
        inverse_indication = items.pop(0)
        inverse = inverse_indication == 'exclude'

        return cls(utterance_idxs=set(items), inverse=inverse)

    @classmethod
    def name(cls):
        return 'matching_utterance_ids'


class Subview(base.CorpusView):
    """
    A subview is a readonly layer representing some subset of a corpus.
    The assets the subview contains are defined by filter criteria.
    Only if an utterance passes all filter criteria it is contained in the subview.

    Args:
        corpus (CorpusView): The corpus this subview is based on.
        filter_criteria (list, FilterCriterion): List of :py:class:`FilterCriterion`

    Example::

        >>> filter = subview.MatchingUtteranceIdxFilter(utterance_idxs=(['utt-1', 'utt-3']))
        >>> corpus = pingu.corpus.load('path/to/corpus')
        >>> corpus.num_utterances
        14
        >>> subset = subview.Subview(self.corpus, filter_criteria=[filter])
        >>> subset.num_utterances
        2
    """

    def __init__(self, corpus, filter_criteria=[]):
        self.corpus = corpus

        if isinstance(filter_criteria, list):
            self.filter_criteria = filter_criteria
        else:
            self.filter_criteria = [filter_criteria]

    @property
    def name(self):
        return 'subview of {}'.format(self.corpus.name)

    @property
    def files(self):
        return {utterance.file.idx: utterance.file for utterance in self.utterances.values()}

    @property
    def utterances(self):
        filtered_utterances = {}

        for utt_idx, utterance in self.corpus.utterances.items():
            matches = True

            for criterion in self.filter_criteria:
                if not criterion.match(utterance, self.corpus):
                    matches = False

            if matches:
                filtered_utterances[utt_idx] = utterance

        return filtered_utterances

    @property
    def issuers(self):
        return {utterance.issuer.idx: utterance.issuer for utterance in self.utterances.values()}

    @property
    def feature_containers(self):
        return self.corpus.feature_containers
