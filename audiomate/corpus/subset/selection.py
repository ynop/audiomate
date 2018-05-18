import collections
import random

from . import subview
from . import utils


class SubsetGenerator(object):
    """
    This class is used to generate subsets of a corpus.

    Args:
        corpus (Corpus): The corpus to create subsets from.
        random_seed (int): Seed to use for random number generation.
    """

    def __init__(self, corpus, random_seed=None):
        self.corpus = corpus
        self.random_seed = random_seed
        self.rand = random.Random()
        self.rand.seed(a=random_seed)

    def random_subset(self, relative_size, balance_labels=False):
        """
        Create a subview of random utterances with a approximate size relative to the full corpus.
        By default x random utterances are selected with x equal to ``relative_size * corpus.num_utterances``.

        Args:
            relative_size (float): A value between 0 and 1.
                                   (0.5 will create a subset with approximately 50% of the full corpus size)
            balance_labels (bool): If True, the labels of the selected utterances are balanced as far as possible.
                                   So the count/duration of every label within the subset is equal.

        Returns:
            Subview: The subview representing the subset.
        """

        num_utterances_in_subset = round(relative_size * self.corpus.num_utterances)
        all_utterance_ids = sorted(list(self.corpus.utterances.keys()))

        if balance_labels:
            all_label_values = self.corpus.all_label_values()
            utterance_with_label_counts = collections.defaultdict(dict)

            for utterance_idx, utterance in self.corpus.utterances.items():
                utterance_with_label_counts[utterance_idx] = utterance.label_count()

            subset_utterance_ids = utils.select_balanced_subset(utterance_with_label_counts,
                                                                num_utterances_in_subset,
                                                                list(all_label_values),
                                                                seed=self.rand.random())

        else:
            subset_utterance_ids = self.rand.sample(all_utterance_ids,
                                                    num_utterances_in_subset)

        filter = subview.MatchingUtteranceIdxFilter(utterance_idxs=set(subset_utterance_ids))
        return subview.Subview(self.corpus, filter_criteria=[filter])

    def random_subset_by_duration(self, relative_duration, balance_labels=False):
        """
        Create a subview of random utterances with a approximate duration relative to the full corpus.
        Random utterances are selected so that the sum of all utterance durations
        equals to the relative duration of the full corpus.

        Args:
            relative_duration (float): A value between 0 and 1. (e.g. 0.5 will create a subset with approximately
                                       50% of the full corpus duration)
            balance_labels (bool): If True, the labels of the selected utterances are balanced as far as possible.
                                   So the count/duration of every label within the subset is equal.

        Returns:
            Subview: The subview representing the subset.
        """
        total_duration = self.corpus.total_duration
        subset_duration = relative_duration * total_duration
        all_label_values = self.corpus.all_label_values()
        utterance_durations = {utt_idx: utt.duration for utt_idx, utt in self.corpus.utterances.items()}

        if balance_labels:
            label_durations = {utt_idx: utt.label_total_duration() for utt_idx, utt in self.corpus.utterances.items()}

            subset_utterance_ids = utils.select_balanced_subset(label_durations,
                                                                subset_duration,
                                                                list(all_label_values),
                                                                select_count_values=utterance_durations,
                                                                seed=self.rand.random())

        else:
            dummy_weights = {utt_idx: {'w': 1} for utt_idx in self.corpus.utterances.keys()}
            subset_utterance_ids = utils.select_balanced_subset(dummy_weights,
                                                                subset_duration,
                                                                ['w'],
                                                                select_count_values=utterance_durations,
                                                                seed=self.rand.random())

        filter = subview.MatchingUtteranceIdxFilter(utterance_idxs=set(subset_utterance_ids))
        return subview.Subview(self.corpus, filter_criteria=[filter])

    def random_subsets(self, relative_sizes, by_duration=False, balance_labels=False):
        """
        Create a bunch of subsets with the given sizes relative to the size or duration of the full corpus.
        Basically the same as calling ``random_subset`` or ``random_subset_by_duration`` multiple times
        with different values. But this method makes sure that every subset contains only utterances,
        that are also contained in the next bigger subset.

        Args:
            relative_sizes (list): A list of numbers between 0 and 1 indicating the sizes of the desired subsets,
                                   relative to the full corpus.
            by_duration (bool): If True the size measure is the duration of all utterances in a subset/corpus.
            balance_labels (bool): If True the labels contained in a subset are chosen to be balanced
                                   as far as possible.

        Returns:
            dict : A dictionary containing all subsets with the relative size as key.
        """
        resulting_sets = {}
        next_bigger_subset = self.corpus

        for relative_size in reversed(relative_sizes):
            generator = SubsetGenerator(next_bigger_subset, random_seed=self.random_seed)

            if by_duration:
                sv = generator.random_subset_by_duration(relative_size, balance_labels=balance_labels)
            else:
                sv = generator.random_subset(relative_size, balance_labels=balance_labels)

            resulting_sets[relative_size] = sv

        return resulting_sets
