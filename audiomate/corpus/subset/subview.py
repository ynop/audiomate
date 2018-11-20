import abc

from audiomate.corpus import base


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
        utterance_idxs (:class:`set`): A list of utterance-ids. Only utterances in the list will pass the
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


class MatchingLabelFilter(FilterCriterion):
    """
    A filter criterion that only accepts utterances which only have the given labels.

    Args:
        labels (:class:`set`): A set of labels which are accepted.
        label_list_ids (:class:`set`): Only check label-lists with these ids. If empty checks all label-lists.
    """

    def __init__(self, labels=set(), label_list_ids=set()):
        self.labels = labels
        self.label_list_ids = label_list_ids

    def match(self, utterance, corpus):
        for label_list_idx, label_list in utterance.label_lists.items():
            if len(self.label_list_ids) == 0 or label_list_idx in self.label_list_ids:
                for label in label_list:
                    if label.value not in self.labels:
                        return False

        return True

    def serialize(self):
        ll_ids = ','.join(sorted(self.label_list_ids))
        labels = ','.join(sorted(self.labels))
        return '{}|||{}'.format(ll_ids, labels)

    @classmethod
    def parse(cls, representation):
        parts = representation.strip().split('|||')

        if len(parts) > 1:
            ll_ids = parts[0].strip().split(',')
            labels = parts[1].strip().split(',')
        else:
            ll_ids = set()
            labels = parts[0].strip().split(',')

        return cls(set(labels), set(ll_ids))

    @classmethod
    def name(cls):
        return 'matching_labels'


__filter_criteria = {}
for cls in FilterCriterion.__subclasses__():
    __filter_criteria[cls.name()] = cls


class UnknownFilterCriteriaException(Exception):
    pass


def available_filter_criteria():
    """
    Get a mapping of all available filter criteria.

    Returns:
        dict: A dictionary with filter-criterion classes with the name of these criteria as key.

    Example::

        >>> available_filter_criteria()
        {
            "matching_utterance_ids" : audiomate.corpus.subview.MatchingUtteranceIdxFilter
        }
    """
    return __filter_criteria


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
        >>> corpus = audiomate.corpus.load('path/to/corpus')
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
    def tracks(self):
        return {utterance.track.idx: utterance.track for utterance in self.utterances.values()}

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
        issuers = {}

        for utterance in self.utterances.values():
            if utterance.issuer is not None:
                issuers[utterance.issuer.idx] = utterance.issuer

        return issuers

    @property
    def feature_containers(self):
        return self.corpus.feature_containers

    def serialize(self):
        """
        Return a string representing the subview with all of its filter criteria.

        Returns:
            str: String with subview definition.
        """
        lines = []

        for criterion in self.filter_criteria:
            lines.append(criterion.name())
            lines.append(criterion.serialize())

        return '\n'.join(lines)

    @classmethod
    def parse(cls, representation, corpus=None):
        """
        Creates a subview from a string representation (created with ``self.serialize``).

        Args:
            representation (str): The representation.

        Returns:
            Subview: The created subview.
        """

        criteria_definitions = representation.split('\n')
        criteria = []

        for i in range(0, len(criteria_definitions), 2):
            filter_name = criteria_definitions[i]
            filter_repr = criteria_definitions[i + 1]

            if filter_name not in available_filter_criteria():
                raise UnknownFilterCriteriaException('Unknown filter-criterion {}'.format(filter_name))

            criterion = available_filter_criteria()[filter_name].parse(filter_repr)
            criteria.append(criterion)

        return cls(corpus, criteria)
