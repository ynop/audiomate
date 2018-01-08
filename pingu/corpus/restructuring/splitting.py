"""
This module contains functionality for splitting a corpus.
"""

import collections
import random

from pingu.corpus import subview


class Splitter(object):
    """
    A splitter provides methods for splitting a corpus into different subsets.
    It provides different approaches for splitting the corpus. (Methods indicated by ``split_by_``)
    These methods mostly take some proportions parameter, which defines how big (in relation) the
    subsets should be. The subsets are returned as :py:class:`pingu.corpus.Subview`.

    Args:
        corpus (Corpus): The corpus that should be splitted.
        random_seed (int): Seed to use for random number generation.
    """

    def __init__(self, corpus, random_seed=None):
        self.corpus = corpus
        self.rand = random.Random()
        self.rand.seed(a=random_seed)

    def split_by_number_of_utterances(self, proportions={}):
        """
        Split the corpus into subsets with the given number of utterances.
        The corpus gets splitted into len(proportions) parts, so the number of utterances are
        distributed according to the proportions.

        Args:
            proportions (dict): A dictionary containing the relative size of the target subsets.
            The key is an identifier for the subset.

        Returns:
            (dict): A dictionary containing the subsets with the identifier from the input as key.

        Example::

            >>> spl = Splitter(corpus)
            >>> corpus.num_utterances
            100
            >>> subsets = spl.split_by_number_of_utterances(proportions={
            >>>     "train" : 0.6,
            >>>     "dev" : 0.2,
            >>>     "test" : 0.2
            >>> })
            >>> print(subsets)
            {'dev': <pingu.corpus.subview.Subview at 0x104ce7400>,
            'test': <pingu.corpus.subview.Subview at 0x104ce74e0>,
            'train': <pingu.corpus.subview.Subview at 0x104ce7438>}
            >>> subsets['train'].num_utterances
            60
            >>> subset['test'].num_utterances
            20
        """

        utterance_idxs = sorted(list(self.corpus.utterances.keys()))
        self.rand.shuffle(utterance_idxs)
        splits = Splitter.split_identifiers(identifiers=utterance_idxs,
                                            proportions=proportions)

        return self._subviews_from_utterance_splits(splits)

    def split_by_proportionally_distribute_labels(self, proportions={}):
        """
        Split the corpus into subsets, so the occurrence of the labels is distributed amongst the
        subsets according to the given proportions.

        Args:
            proportions (dict): A dictionary containing the relative size of the target subsets.
            The key is an identifier for the subset.

        Returns:
            (dict): A dictionary containing the subsets with the identifier from the input as key.
        """

        identifiers = {}

        for utterance in self.corpus.utterances.values():
            identifiers[utterance.idx] = utterance.label_count()

        splits = Splitter.get_identifiers_splitted_by_weights(identifiers, proportions)

        return self._subviews_from_utterance_splits(splits)

    def _subviews_from_utterance_splits(self, splits):
        """
        Create subviews from a dict containing utterance-ids for each subview.

        e.g. {'train': ['utt-1', 'utt-2'], 'test': [...], ...}
        """
        subviews = {}

        for idx, subview_utterances in splits.items():
            filter = subview.MatchingUtteranceIdxFilter(utterance_idxs=subview_utterances)
            split = subview.Subview(self.corpus, filter_criteria=filter)
            subviews[idx] = split

        return subviews

    @staticmethod
    def split_identifiers(identifiers=[], proportions={}):
        """
        Split the given identifiers by the given proportions.

        Args:
            identifiers (list): List of identifiers (str).
            proportions (dict): A dictionary containing the proportions with the identifier from the
            input as key.

        Returns:
            dict: Dictionary containing a list of identifiers per part with the same key as the
            proportions dict.

        Example::

            >>> Splitter.split_identifiers(
            >>>     identifiers=['a', 'b', 'c', 'd'],
            >>>     proportions={'melvin' : 0.5, 'timmy' : 0.5}
            >>> )
            {'melvin' : ['a', 'c'], 'timmy' : ['b', 'd']}
        """

        absolute_proportions = Splitter.absolute_proportions(proportions, len(identifiers))

        parts = {}
        start_index = 0

        for idx, proportion in absolute_proportions.items():
            parts[idx] = identifiers[start_index:start_index + proportion]
            start_index += proportion

        return parts

    @staticmethod
    def absolute_proportions(proportions, count):
        """
        Split a given integer into n parts according to len(proportions) so they sum up to count and
        match the given proportions.

        Args:
            proportions (dict): Dict of proportions, with a identifier as key.

        Returns:
            dict: Dictionary with absolute proportions and same identifiers as key.

        Example::

            >>> Splitter.absolute_proportions({'train': 0.5, 'test': 0.5}, 100)
            {'train': 50, 'test': 50}
        """

        # first create absolute values by flooring non-integer portions
        relative_sum = sum(proportions.values())
        absolute_proportions = {idx: int(count / relative_sum * prop_value) for idx, prop_value in
                                proportions.items()}

        # Now distribute the rest value randomly over the different parts
        absolute_sum = sum(absolute_proportions.values())
        rest_value = count - absolute_sum
        subset_keys = sorted(list(proportions.keys()))

        for i in range(rest_value):
            key = subset_keys[i % len(subset_keys)]
            absolute_proportions[key] += 1

        return absolute_proportions

    @staticmethod
    def get_identifiers_splitted_by_weights(identifiers={}, proportions={}):
        """
        Divide the given identifiers based on the given proportions. But instead of randomly split
        the identifiers it is based on category weights. Every identifier has a weight for any
        number of categories. The target is, to split the identifiers in a way, so the sum of
        category k within part x is proportional to the sum of category x over all parts
        according to the given proportions. This is done by greedily insert the identifiers step by
        step in a part which has free space (weight). If there are no fitting parts anymore, the one
        with the least weight exceed is used.

        Args:
            identifiers (dict): A dictionary containing the weights for each identifier (key). Per
                                item a dictionary of weights per category is given.
            proportions (dict): Dict of proportions, with a identifier as key.

        Returns:
            dictionary (dict): Dictionary containing a list of identifiers per part with the same
                               key as the proportions dict.

        Example::

            >>> identifiers = {
            >>>     'a': {'music': 2, 'speech': 1},
            >>>     'b': {'music': 5, 'speech': 2},
            >>>     'c': {'music': 2, 'speech': 4},
            >>>     'd': {'music': 1, 'speech': 4},
            >>>     'e': {'music': 3, 'speech': 4}
            >>> }
            >>> proportions = {
            >>>     "train" : 0.6,
            >>>     "dev" : 0.2,
            >>>     "test" : 0.2
            >>> }
            >>> Splitter.get_identifiers_splitted_by_weights(identifiers, proportions)
            {
                'train': ['a', 'b', 'd'],
                'dev': ['c'],
                'test': ['e']
            }
        """

        # Get total weight per category
        sum_per_category = collections.defaultdict(int)

        for identifier, cat_weights in identifiers.items():
            for category, weight in cat_weights.items():
                sum_per_category[category] += weight

        target_weights_per_part = collections.defaultdict(dict)

        # Get target weight for each part and category
        for category, total_weight in sum_per_category.items():
            absolute_proportions = Splitter.absolute_proportions(proportions, total_weight)

            for idx, proportion in absolute_proportions.items():
                target_weights_per_part[idx][category] = proportion

        # Distribute items greedily
        part_ids = sorted(list(proportions.keys()))
        current_weights_per_part = {idx: collections.defaultdict(int) for idx in part_ids}
        result = collections.defaultdict(list)

        for identifier in sorted(identifiers.keys()):
            cat_weights = identifiers[identifier]

            target_part = None
            current_part = 0
            weight_over_target = collections.defaultdict(int)

            # Search for fitting part
            while target_part is None and current_part < len(part_ids):
                free_space = True
                part_id = part_ids[current_part]
                part_weights = current_weights_per_part[part_id]

                for category, weight in cat_weights.items():
                    target_weight = target_weights_per_part[part_id][category]
                    current_weight = part_weights[category]
                    weight_diff = current_weight + weight - target_weight
                    weight_over_target[part_id] += weight_diff

                    if weight_diff > 0:
                        free_space = False

                # If weight doesn't exceed target, place identifier in part
                if free_space:
                    target_part = part_id

                current_part += 1

            # If not found fitting part, select the part with the least overweight
            if target_part is None:
                target_part = sorted(weight_over_target.items(), key=lambda x: x[1])[0][0]

            result[target_part].append(identifier)

            for category, weight in cat_weights.items():
                current_weights_per_part[target_part][category] += weight

        return result
