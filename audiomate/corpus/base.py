import abc
import collections

import numpy as np

from audiomate.utils import stats


class CorpusView(metaclass=abc.ABCMeta):
    """
    This class defines the basic interface of a corpus. It is not meant to be instantiated directly.
    It only describes the methods for accessing data of the corpus.

    Notes:
        All paths to files should be held as absolute paths in memory.
    """

    @property
    @abc.abstractmethod
    def name(self):
        """ Return the name of the dataset (Equals basename of the path, if not None). """
        return 'undefined'

    #
    #   Tracks
    #

    @property
    @abc.abstractmethod
    def tracks(self):
        """
        Return the tracks in the corpus.

        Returns:
            dict: A dictionary containing :py:class:`audiomate.track.Track` objects with the
            track-idx as key.
        """
        return {}

    @property
    def num_tracks(self):
        """ Return number of tracks. """
        return len(self.tracks)

    #
    #   Utterances
    #

    @property
    @abc.abstractmethod
    def utterances(self):
        """
        Return the utterances in the corpus.

        Returns:
            dict: A dictionary containing :py:class:`audiomate.corpus.assets.Utterance` objects with the
            utterance-idx as key.
        """
        return {}

    @property
    def num_utterances(self):
        """ Return number of utterances. """
        return len(self.utterances)

    #
    #   Issuers
    #

    @property
    @abc.abstractmethod
    def issuers(self):
        """
        Return the issuers in the corpus.

        Returns:
            dict: A dictionary containing :py:class:`audiomate.issuers.Issuer` objects with the
            issuer-idx as key.
        """
        return {}

    @property
    def num_issuers(self):
        """ Return the number of issuers in the corpus. """
        return len(self.issuers)

    #
    #   Feature Container
    #

    @property
    @abc.abstractmethod
    def feature_containers(self):
        """
        Return the feature-containers in the corpus.

        Returns:
            dict: A dictionary containing :py:class:`audiomate.container.FeatureContainer` objects
            with the feature-idx as key.
        """
        return {}

    @property
    def num_feature_containers(self):
        """ Return the number of feature-containers in the corpus. """
        return len(self.feature_containers)

    #
    #   Subviews
    #

    @property
    def subviews(self):
        """
        Return the subviews of the corpus.

        Returns:
             dict: A dictionary containing :py:class:`audiomate.corpus.Subview` objects with the subview-idx as key.
        """
        return {}

    @property
    def num_subviews(self):
        """ Return the number of subviews in the corpus. """
        return len(self.subviews)

    #
    #   Labels
    #

    def all_label_values(self, label_list_ids=None):
        """
        Return a set of all label-values occurring in this corpus.

        Args:
            label_list_ids (list): If not None, only labels from label-lists with an id contained in this list
                                   are considered.

        Returns:
             :class:`set`: A set of distinct label-values.
        """
        values = set()

        for utterance in self.utterances.values():
            values = values.union(utterance.all_label_values(label_list_ids=label_list_ids))

        return values

    def label_count(self, label_list_ids=None):
        """
        Return a dictionary containing the number of times, every label-value in this corpus is occurring.

        Args:
            label_list_ids (list): If not None, only labels from label-lists with an id contained in this list
                                   are considered.

        Returns:
            dict: A dictionary containing the number of occurrences with the label-value as key.
        """
        count = collections.defaultdict(int)

        for utterance in self.utterances.values():
            for label_value, utt_count in utterance.label_count(label_list_ids=label_list_ids).items():
                count[label_value] += utt_count

        return count

    def label_durations(self, label_list_ids=None):
        """
        Return a dictionary containing the total duration, every label-value in this corpus is occurring.

        Args:
            label_list_ids (list): If not None, only labels from label-lists with an id contained in this list
                                   are considered.

        Returns:
            dict: A dictionary containing the total duration with the label-value as key.
        """
        duration = collections.defaultdict(int)

        for utterance in self.utterances.values():
            for label_value, utt_count in utterance.label_total_duration(label_list_ids=label_list_ids).items():
                duration[label_value] += utt_count

        return duration

    def all_tokens(self, delimiter=' ', label_list_ids=None):
        """
        Return a list of all tokens occurring in one of the labels in the corpus.

        Args:
            delimiter (str): The delimiter used to split labels into tokens
                             (see :meth:`audiomate.annotations.Label.tokenized`).
            label_list_ids (list): If not None, only labels from label-lists with an idx contained in this list
                                   are considered.

        Returns:
             :class:`set`: A set of distinct tokens.
        """
        tokens = set()

        for utterance in self.utterances.values():
            tokens = tokens.union(utterance.all_tokens(delimiter=delimiter, label_list_ids=label_list_ids))

        return tokens

    #
    #   Data
    #

    @property
    def total_duration(self):
        """
        Return the total amount of audio summed over all utterances in the corpus in seconds.
        """
        duration = 0

        for utterance in self.utterances.values():
            duration += utterance.duration

        return duration

    def stats(self):
        """
        Return statistics calculated overall samples of all utterances in the corpus.

        Returns:
            DataStats: A DataStats object containing statistics overall samples in the corpus.
        """

        per_utt_stats = self.stats_per_utterance()
        return stats.DataStats.concatenate(per_utt_stats.values())

    def stats_per_utterance(self):
        """
        Return statistics calculated for all samples of each utterance in the corpus.

        Returns:
            dict: A dictionary containing a DataStats object for each utt.
        """

        all_stats = {}

        for utterance in self.utterances.values():
            data = utterance.read_samples()
            all_stats[utterance.idx] = stats.DataStats(float(np.mean(data)),
                                                       float(np.var(data)),
                                                       np.min(data),
                                                       np.max(data),
                                                       data.size)

        return all_stats
