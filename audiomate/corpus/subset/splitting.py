import random
import collections

from audiomate.corpus.subset import subview
from audiomate.corpus.subset import utils


class Splitter:
    """
    A splitter provides methods for splitting a corpus into different subsets.
    It provides different approaches for splitting the corpus.
    (Methods indicated by ``split_by_``)
    These methods mostly take some proportions parameter,
    which defines how big (in relation) the
    subsets should be. The subsets are returned
    as :py:class:`audiomate.corpus.Subview`.

    Args:
        corpus (Corpus): The corpus that should be splitted.
        random_seed (int): Seed to use for random number generation.
    """

    def __init__(self, corpus, random_seed=None):
        self.corpus = corpus
        self.rand = random.Random()
        self.rand.seed(a=random_seed)

    def split(self, proportions, separate_issuers=False):
        """
        Split the corpus based on the number of utterances.
        The utterances are distributed to `len(proportions)` subsets,
        according to the ratios `proportions[subset]`.

        Args:
            proportions (dict): A dictionary containing the relative size of
                                the target subsets. The key is an identifier
                                for the subset.
            separate_issuers (bool): If True it makes sure that all utterances
                                     of an issuer are in the same subset.

        Returns:
            (dict): A dictionary containing the subsets with the identifier
                    from the input as key.

        Example::

            >>> spl = Splitter(corpus)
            >>> corpus.num_utterances
            100
            >>> subsets = spl.split(proportions={
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
            >>> subsets['dev'].num_utterances
            20
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

            issuer_utt_count = {
                k: {'count': int(v)}
                for k, v in issuer_utt_count.items()
            }

            # Split with total utt count per issuer as weight
            issuer_splits = utils.get_identifiers_splitted_by_weights(
                issuer_utt_count,
                proportions,
                seed=self.rand.random()
            )

            return self._subviews_from_issuer_splits(issuer_splits, issuer_utts)

        else:
            utterance_idxs = list(self.corpus.utterances.keys())
            splits = utils.split_identifiers(
                utterance_idxs,
                proportions,
                seed=self.rand.random()
            )

            return self._subviews_from_utterance_splits(splits)

    def split_by_audio_duration(self, proportions, separate_issuers=False):
        """
        Split the corpus based on the the total duration of audio.
        The utterances are distributed to `len(proportions)` subsets.
        Utterances are split up in a way that each subset contains
        audio with a duration proportional to the given proportions.

        Args:
            proportions (dict): A dictionary containing the relative size of
                                the target subsets. The key is an identifier
                                for the subset.
            separate_issuers (bool): If True it makes sure that all utterances
                                     of an issuer are in the same subset.

        Returns:
            (dict): A dictionary containing the subsets with the identifier
                    from the input as key.

        Example::

            >>> spl = Splitter(corpus)
            >>> corpus.num_utterances
            100
            >>> subsets = spl.split_by_audio_duration(proportions={
            >>>     "train" : 0.6,
            >>>     "dev" : 0.2,
            >>>     "test" : 0.2
            >>> })
            >>> print(subsets)
            {'dev': <audiomate.corpus.subview.Subview at 0x104ce7400>,
            'test': <audiomate.corpus.subview.Subview at 0x104ce74e0>,
            'train': <audiomate.corpus.subview.Subview at 0x104ce7438>}
            >>> subsets['train'].num_utterances
            55
            >>> subsets['dev'].num_utterances
            35
            >>> subsets['test'].num_utterances
            10
        """

        if separate_issuers:
            # Count total length of utterances per issuer
            issuer_utts_duration = collections.defaultdict(float)
            issuer_utts = collections.defaultdict(list)

            for utterance in self.corpus.utterances.values():
                issuer_utts_duration[utterance.issuer.idx] += utterance.duration
                issuer_utts[utterance.issuer.idx].append(utterance.idx)

            issuer_utts_duration = {
                k: {'duration': int(v * 1000)}
                for k, v in issuer_utts_duration.items()
            }

            # Split with total utt duration per issuer as weight
            issuer_splits = utils.get_identifiers_splitted_by_weights(
                issuer_utts_duration,
                proportions,
                seed=self.rand.random()
            )

            return self._subviews_from_issuer_splits(issuer_splits, issuer_utts)

        else:
            utterance_to_duration = {}

            for utterance in self.corpus.utterances.values():
                utterance_to_duration[utterance.idx] = {
                    'duration': int(utterance.duration * 1000)
                }

            splits = utils.get_identifiers_splitted_by_weights(
                utterance_to_duration,
                proportions,
                seed=self.rand.random()
            )

            return self._subviews_from_utterance_splits(splits)

    def split_by_label_length(self, proportions,
                              label_list_idx=None,
                              separate_issuers=False):
        """
        Split the corpus based on the the total length of the label-list.
        The utterances are distributed to `len(proportions)` subsets.
        Utterances are split up in a way that each subset contains
        labels summed up to a length proportional to the given proportions.
        Length is defined as the number of characters.

        Args:
            proportions (dict): A dictionary containing the relative size
                                of the target subsets.
                                The key is an identifier for the subset.
            label_list_idx (str): The idx of the label-list to use for compute
                                  the length. If `None` all label-lists are used.
            separate_issuers (bool): If True it makes sure that all utterances
                                     of an issuer are in the same subset.

        Returns:
            (dict): A dictionary containing the subsets with the identifier
                    from the input as key.

        Example::

            >>> spl = Splitter(corpus)
            >>> corpus.num_utterances
            100
            >>> subsets = spl.split_by_label_length(proportions={
            >>>     "train" : 0.6,
            >>>     "dev" : 0.2,
            >>>     "test" : 0.2
            >>> })
            >>> print(subsets)
            {'dev': <audiomate.corpus.subview.Subview at 0x104ce7400>,
            'test': <audiomate.corpus.subview.Subview at 0x104ce74e0>,
            'train': <audiomate.corpus.subview.Subview at 0x104ce7438>}
            >>> subsets['train'].num_utterances
            55
            >>> subsets['dev'].num_utterances
            35
            >>> subsets['test'].num_utterances
            10
        """

        if separate_issuers:
            # Count total length of utterances per issuer
            issuer_utts_length = collections.defaultdict(int)
            issuer_utts = collections.defaultdict(list)

            for utterance in self.corpus.utterances.values():
                lls = utterance.label_lists

                if label_list_idx is None:
                    num_char = sum(ll.total_length for ll in lls.values())
                else:
                    num_char = lls[label_list_idx].total_length

                issuer_utts_length[utterance.issuer.idx] += num_char
                issuer_utts[utterance.issuer.idx].append(utterance.idx)

            issuer_utts_length = {
                k: {'length': v}
                for k, v in issuer_utts_length.items()
            }

            # Split with total utt duration per issuer as weight
            issuer_splits = utils.get_identifiers_splitted_by_weights(
                issuer_utts_length,
                proportions,
                seed=self.rand.random()
            )

            return self._subviews_from_issuer_splits(issuer_splits, issuer_utts)

        else:
            utterance_to_length = {}

            for utterance in self.corpus.utterances.values():
                lls = utterance.label_lists

                if label_list_idx is None:
                    num_char = sum(ll.total_length for ll in lls.values())
                else:
                    num_char = lls[label_list_idx].total_length

                utterance_to_length[utterance.idx] = {
                    'length': num_char
                }

            splits = utils.get_identifiers_splitted_by_weights(
                utterance_to_length,
                proportions,
                seed=self.rand.random()
            )

            return self._subviews_from_utterance_splits(splits)

    def split_by_label_occurence(self, proportions, separate_issuers=False):
        """
        Split the corpus based on the total number of occcurences of labels.
        The utterances are distributed to `len(proportions)` subsets.
        Utterances are split up in a way that each subset contains
        labels-occurences proportional to the given proportions.

        Args:
            proportions (dict): A dictionary containing the relative size
                                of the target subsets.
                                The key is an identifier for the subset.
            separate_issuers (bool): If True it makes sure that all utterances
                                     of an issuer are in the same subset.

        Returns:
            (dict): A dictionary containing the subsets with the identifier
                    from the input as key.

        Example::

            >>> spl = Splitter(corpus)
            >>> corpus.num_utterances
            100
            >>> subsets = spl.split_by_label_occurence(proportions={
            >>>     "train" : 0.6,
            >>>     "dev" : 0.2,
            >>>     "test" : 0.2
            >>> })
            >>> print(subsets)
            {'dev': <audiomate.corpus.subview.Subview at 0x104ce7400>,
            'test': <audiomate.corpus.subview.Subview at 0x104ce74e0>,
            'train': <audiomate.corpus.subview.Subview at 0x104ce7438>}
            >>> subsets['train'].num_utterances
            55
            >>> subsets['dev'].num_utterances
            35
            >>> subsets['test'].num_utterances
            10
        """

        if separate_issuers:
            # Count total length of utterances per issuer
            issuer_label_count = collections.defaultdict(collections.Counter)
            issuer_utts = collections.defaultdict(list)

            for utterance in self.corpus.utterances.values():
                label_count = utterance.label_count()
                issuer_label_count[utterance.issuer.idx].update(label_count)
                issuer_utts[utterance.issuer.idx].append(utterance.idx)

            # issuer_label_count = {
            #     k: dict(v) for k, v in issuer_label_count.items()
            # }

            # Split with total utt duration per issuer as weight
            issuer_splits = utils.get_identifiers_splitted_by_weights(
                issuer_label_count,
                proportions,
                seed=self.rand.random()
            )

            return self._subviews_from_issuer_splits(issuer_splits, issuer_utts)

        else:
            utterance_label_count = {
                utt.idx: dict(utt.label_count())
                for utt in self.corpus.utterances.values()
            }

            splits = utils.get_identifiers_splitted_by_weights(
                utterance_label_count,
                proportions,
                seed=self.rand.random()
            )

            return self._subviews_from_utterance_splits(splits)

    def split_by_label_duration(self, proportions, separate_issuers=False):
        """
        Split the corpus based on the total duration of labels (end - start).
        The utterances are distributed to `len(proportions)` subsets.
        Utterances are split up in a way that each subset contains
        labels with a duration proportional to the given proportions.

        Args:
            proportions (dict): A dictionary containing the relative size of
                                the target subsets. The key is an identifier
                                for the subset.
            separate_issuers (bool): If True it makes sure that all utterances
                                     of an issuer are in the same subset.

        Returns:
            (dict): A dictionary containing the subsets with the identifier
                    from the input as key.

        Example::

            >>> spl = Splitter(corpus)
            >>> corpus.num_utterances
            100
            >>> subsets = spl.split_by_label_duration(proportions={
            >>>     "train" : 0.6,
            >>>     "dev" : 0.2,
            >>>     "test" : 0.2
            >>> })
            >>> print(subsets)
            {'dev': <audiomate.corpus.subview.Subview at 0x104ce7400>,
            'test': <audiomate.corpus.subview.Subview at 0x104ce74e0>,
            'train': <audiomate.corpus.subview.Subview at 0x104ce7438>}
            >>> subsets['train'].num_utterances
            55
            >>> subsets['dev'].num_utterances
            35
            >>> subsets['test'].num_utterances
            10
        """

        if separate_issuers:
            # Count total length of utterances per issuer
            issuer_label_duration = collections.defaultdict(collections.Counter)
            issuer_utts = collections.defaultdict(list)

            for utterance in self.corpus.utterances.values():
                issuer_label_duration[utterance.issuer.idx].update(
                    utterance.label_total_duration()
                )
                issuer_utts[utterance.issuer.idx].append(utterance.idx)

            # Split with total utt duration per issuer as weight
            issuer_splits = utils.get_identifiers_splitted_by_weights(
                issuer_label_duration,
                proportions,
                seed=self.rand.random()
            )

            return self._subviews_from_issuer_splits(issuer_splits, issuer_utts)

        else:
            utterance_label_duration = {
                utt.idx: dict(utt.label_total_duration())
                for utt in self.corpus.utterances.values()
            }

            splits = utils.get_identifiers_splitted_by_weights(
                utterance_label_duration,
                proportions,
                seed=self.rand.random()
            )

            return self._subviews_from_utterance_splits(splits)

    def _subviews_from_utterance_splits(self, splits):
        """
        Create subviews from a dict containing utterance-ids for each subview.

        e.g. {'train': ['utt-1', 'utt-2'], 'test': [...], ...}
        """
        subviews = {}

        for idx, subview_utterances in splits.items():
            utt_filter = subview.MatchingUtteranceIdxFilter(utterance_idxs=subview_utterances)
            split = subview.Subview(self.corpus, filter_criteria=utt_filter)
            subviews[idx] = split

        return subviews

    def _subviews_from_issuer_splits(self, splits, issuer_to_utt):
        """
        Create subviews from a dict containing issuer-ids for each subview
        and a map from issuer-ids to utterance-ids.

        e.g. {'train': ['issuer-1', 'issuer-2'], 'test': [...], ...}
        """

        # Collect utterances of all issuers per split
        utt_splits = collections.defaultdict(list)

        for split_idx, issuer_ids in splits.items():
            for issuer_idx in issuer_ids:
                utt_splits[split_idx].extend(issuer_to_utt[issuer_idx])

        return self._subviews_from_utterance_splits(utt_splits)
