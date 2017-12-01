"""
This module contains functionality for splitting a corpus.
"""

import random

from . import subview


class Splitter(object):
    """
    A splitter provides different methods for splitting a corpus into different subsets.

    Args:
        corpus (Corpus): The corpus that should be splitted.
    """

    def __init__(self, corpus):
        self.corpus = corpus

    def split_by_number_of_utterances(self, proportions={}):
        """
        Split the corpus into subsets with the given number of utterances.

        Args:
            proportions (dict): A dictionary containing the relative size of the target subsets. The key is an identifier for the subset.

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

        utterance_idxs = list(self.corpus.utterances.keys())
        splits = Splitter.get_identifiers_randomly_splitted(identifiers=utterance_idxs, proportions=proportions)
        subviews = {}

        for idx, subview_utterances in splits.items():
            filter = subview.MatchingUtteranceIdxFilter(utterance_idxs=subview_utterances)
            split = subview.Subview(self.corpus, filter_criteria=filter)
            subviews[idx] = split

        return subviews

    @staticmethod
    def get_identifiers_randomly_splitted(identifiers=[], proportions={}):
        """
        Split the given identifiers by the given proportions.

        Args:
            identifiers (list): List of identifiers (str).
            proportions (dict): A dictionary containing the proportions with the identifier from the input as key.

        Returns:
            dict: Dictionary containing a list of identifiers per part with the same key as the proportions dict.

        Example::

            >>> Splitter.get_identifiers_randomly_splitted(['a', 'b', 'c', 'd'], proportions={'melvin' : 0.5, 'timmy' : 0.5})
            {'melvin' : ['a', 'c'], 'timmy' : ['b', 'd']}
        """

        absolute_proportions = Splitter.absolute_proportions(proportions, len(identifiers))

        random.shuffle(identifiers)

        parts = {}
        start_index = 0

        for idx, proportion in absolute_proportions.items():
            parts[idx] = identifiers[start_index:start_index + proportion]
            start_index += proportion

        return parts

    @staticmethod
    def absolute_proportions(proportions, count):
        """
        Split a given integer into n parts according to len(proportions) so they sum up to count and match the given proportions.

        Args:
            proportions (dict): Dict of proportions, with a identifier as key.

        Returns:
            dict: Dictionary with absolute proportions and same identifiers as key.
        """

        # first create absolute values by flooring non-integer portions
        relative_sum = sum(proportions.values())
        absolute_proportions = {idx: int(count / relative_sum * prop_value) for idx, prop_value in proportions.items()}

        # Now distribute the rest value randomly over the different parts
        absolute_sum = sum(absolute_proportions.values())
        rest_value = count - absolute_sum
        subset_keys = list(proportions.keys())

        for i in range(rest_value):
            key = subset_keys[i % len(subset_keys)]
            absolute_proportions[key] += 1

        return absolute_proportions
