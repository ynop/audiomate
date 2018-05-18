import random
import collections

from audiomate.corpus.subset import subview
from audiomate.corpus.subset import utils


class Splitter(object):
    """
    A splitter provides methods for splitting a corpus into different subsets.
    It provides different approaches for splitting the corpus. (Methods indicated by ``split_by_``)
    These methods mostly take some proportions parameter, which defines how big (in relation) the
    subsets should be. The subsets are returned as :py:class:`audiomate.corpus.Subview`.

    Args:
        corpus (Corpus): The corpus that should be splitted.
        random_seed (int): Seed to use for random number generation.
    """

    def __init__(self, corpus, random_seed=None):
        self.corpus = corpus
        self.rand = random.Random()
        self.rand.seed(a=random_seed)

    def split_by_length_of_utterances(self, proportions={}, separate_issuers=False):
        """
        Split the corpus into subsets where the total duration of subsets are proportional to the given proportions.
        The corpus gets splitted into len(proportions) parts, so the number of utterances are
        distributed according to the proportions.

        Args:
            proportions (dict): A dictionary containing the relative size of the target subsets.
                                The key is an identifier for the subset.
            separate_issuers (bool): If True it makes sure that all utterances of an issuer are in the same subset.

        Returns:
            (dict): A dictionary containing the subsets with the identifier from the input as key.

        Example::

            >>> spl = Splitter(corpus)
            >>> corpus.num_utterances
            100
            >>> subsets = spl.split_by_length_of_utterances(proportions={
            >>>     "train" : 0.6,
            >>>     "dev" : 0.2,
            >>>     "test" : 0.2
            >>> })
            >>> print(subsets)
            {'dev': <audiomate.corpus.subview.Subview at 0x104ce7400>,
            'test': <audiomate.corpus.subview.Subview at 0x104ce74e0>,
            'train': <audiomate.corpus.subview.Subview at 0x104ce7438>}
            >>> subsets['train'].num_utterances
            60
            >>> subsets['test'].num_utterances
            20
        """

        utterance_to_duration = {}

        if separate_issuers:
            # Count total length of utterances per issuer
            issuer_utts_total_duration = collections.defaultdict(float)
            issuer_utts = collections.defaultdict(list)

            for utterance in self.corpus.utterances.values():
                issuer_utts_total_duration[utterance.issuer.idx] += utterance.duration
                issuer_utts[utterance.issuer.idx].append(utterance.idx)

            issuer_utts_total_duration = {k: {'duration': int(v)} for k, v in issuer_utts_total_duration.items()}

            # Split with total utt duration per issuer as weight
            issuer_splits = utils.get_identifiers_splitted_by_weights(issuer_utts_total_duration,
                                                                      proportions=proportions)

            # Collect utterances of all issuers per split
            splits = collections.defaultdict(list)

            for split_idx, issuer_ids in issuer_splits.items():
                for issuer_idx in issuer_ids:
                    splits[split_idx].extend(issuer_utts[issuer_idx])
        else:
            for utterance in self.corpus.utterances.values():
                utterance_to_duration[utterance.idx] = {'length': int(utterance.duration * 100)}

            splits = utils.get_identifiers_splitted_by_weights(utterance_to_duration, proportions=proportions)

        return self._subviews_from_utterance_splits(splits)

    def split_by_number_of_utterances(self, proportions={}, separate_issuers=False):
        """
        Split the corpus into subsets with the given number of utterances.
        The corpus gets splitted into len(proportions) parts, so the number of utterances are
        distributed according to the proportions.

        Args:
            proportions (dict): A dictionary containing the relative size of the target subsets.
                                The key is an identifier for the subset.
            separate_issuers (bool): If True it makes sure that all utterances of an issuer are in the same subset.

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
            {'dev': <audiomate.corpus.subview.Subview at 0x104ce7400>,
            'test': <audiomate.corpus.subview.Subview at 0x104ce74e0>,
            'train': <audiomate.corpus.subview.Subview at 0x104ce7438>}
            >>> subsets['train'].num_utterances
            60
            >>> subsets['test'].num_utterances
            20
        """

        if separate_issuers:
            # Count number of utterances per issuer
            issuer_utt_count = collections.defaultdict(int)
            issuer_utts = collections.defaultdict(list)

            for utterance in self.corpus.utterances.values():
                issuer_utt_count[utterance.issuer.idx] += 1
                issuer_utts[utterance.issuer.idx].append(utterance.idx)

            issuer_utt_count = {k: {'count': int(v)} for k, v in issuer_utt_count.items()}

            # Split with total utt duration per issuer as weight
            issuer_splits = utils.get_identifiers_splitted_by_weights(issuer_utt_count,
                                                                      proportions=proportions)

            # Collect utterances of all issuers per split
            splits = collections.defaultdict(list)

            for split_idx, issuer_ids in issuer_splits.items():
                for issuer_idx in issuer_ids:
                    splits[split_idx].extend(issuer_utts[issuer_idx])
        else:
            utterance_idxs = sorted(list(self.corpus.utterances.keys()))
            self.rand.shuffle(utterance_idxs)
            splits = utils.split_identifiers(identifiers=utterance_idxs,
                                             proportions=proportions)

        return self._subviews_from_utterance_splits(splits)

    def split_by_proportionally_distribute_labels(self, proportions={}, use_lengths=True):
        """
        Split the corpus into subsets, so the occurrence of the labels is distributed amongst the
        subsets according to the given proportions.

        Args:
            proportions (dict): A dictionary containing the relative size of the target subsets.
                                The key is an identifier for the subset.
            use_lengths (bool): If True the lengths of the labels are considered for splitting proportionally,
                                otherwise only the number of occurrences is taken into account.

        Returns:
            (dict): A dictionary containing the subsets with the identifier from the input as key.
        """

        identifiers = {}

        for utterance in self.corpus.utterances.values():
            if use_lengths:
                identifiers[utterance.idx] = {l: int(d * 100) for l, d in utterance.label_total_duration().items()}
            else:
                identifiers[utterance.idx] = utterance.label_count()

        splits = utils.get_identifiers_splitted_by_weights(identifiers, proportions)

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
