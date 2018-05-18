import unittest
import collections

import pytest

from audiomate.corpus.subset import splitting
from tests import resources


class SplitterTest(unittest.TestCase):
    def setUp(self):
        self.corpus = resources.create_multi_label_corpus()
        self.splitter = splitting.Splitter(self.corpus)

    def test_split_by_length_of_utterances(self):
        res = self.splitter.split_by_length_of_utterances({
            'train': 0.6,
            'test': 0.2
        })

        train_utt_ids = res['train'].utterances.keys()
        test_utt_ids = res['test'].utterances.keys()

        train_duration = sum([utt.duration for utt in res['train'].utterances.values()])
        test_duration = sum([utt.duration for utt in res['test'].utterances.values()])

        assert set(train_utt_ids).union(test_utt_ids) == set(self.corpus.utterances.keys())
        assert train_duration / test_duration == pytest.approx(3, rel=0.1)

    def test_split_by_length_of_utterances_issuer_separated(self):
        res = self.splitter.split_by_length_of_utterances({
            'train': 1.0,
            'test': 1.2
        }, separate_issuers=True)

        train_utt_ids = res['train'].utterances.keys()
        test_utt_ids = res['test'].utterances.keys()

        train_duration = sum([utt.duration for utt in res['train'].utterances.values()])
        test_duration = sum([utt.duration for utt in res['test'].utterances.values()])

        subsets_of_issuers = collections.defaultdict(set)

        for utt in res['train'].utterances.values():
            subsets_of_issuers[utt.issuer.idx].add('train')

        for utt in res['test'].utterances.values():
            subsets_of_issuers[utt.issuer.idx].add('test')

        assert set(train_utt_ids).union(test_utt_ids) == set(self.corpus.utterances.keys())
        assert train_duration / test_duration == pytest.approx(1.0/1.2, rel=0.1)

        for issuer_idx, subset_list in subsets_of_issuers.items():
            assert len(subset_list) == 1

    def test_split_by_number_of_utterances(self):
        res = self.splitter.split_by_number_of_utterances({
            'train': 0.6,
            'test': 0.2
        })

        self.assertEqual(6, res['train'].num_utterances)
        self.assertEqual(2, res['test'].num_utterances)

    def test_split_by_number_of_utterances_issuer_separated(self):
        res = self.splitter.split_by_number_of_utterances({
            'train': 0.6,
            'test': 0.2
        }, separate_issuers=True)

        subsets_of_issuers = collections.defaultdict(set)

        for utt in res['train'].utterances.values():
            subsets_of_issuers[utt.issuer.idx].add('train')

        for utt in res['test'].utterances.values():
            subsets_of_issuers[utt.issuer.idx].add('test')

        print(subsets_of_issuers)

        self.assertEqual(6, res['train'].num_utterances)
        self.assertEqual(2, res['test'].num_utterances)

        for issuer_idx, subset_list in subsets_of_issuers.items():
            self.assertEqual(1, len(subset_list))

    def test_split_by_number_of_utterances_seed(self):
        self.corpus = resources.create_multi_label_corpus()
        res1 = splitting.Splitter(self.corpus, random_seed=15).split_by_number_of_utterances({
            'train': 0.6,
            'test': 0.2
        })

        self.corpus = resources.create_multi_label_corpus()
        res2 = splitting.Splitter(self.corpus, random_seed=15).split_by_number_of_utterances({
            'train': 0.6,
            'test': 0.2
        })

        self.assertSetEqual(set(res1['train'].utterances.keys()),
                            set(res2['train'].utterances.keys()))
        self.assertSetEqual(set(res1['test'].utterances.keys()),
                            set(res2['test'].utterances.keys()))

    def test_split_by_proportionally_distribute_labels_by_lengths(self):
        res = self.splitter.split_by_proportionally_distribute_labels({
            'train': 0.6,
            'test': 0.2
        })

        train_utt_ids = res['train'].utterances.keys()
        test_utt_ids = res['test'].utterances.keys()

        train_duration = sum([utt.duration for utt in res['train'].utterances.values()])
        test_duration = sum([utt.duration for utt in res['test'].utterances.values()])

        assert set(train_utt_ids).union(test_utt_ids) == set(self.corpus.utterances.keys())
        assert train_duration / test_duration == pytest.approx(3, rel=0.3)

    def test_split_by_proportionally_distribute_labels_by_number(self):
        res = self.splitter.split_by_proportionally_distribute_labels({
            'train': 0.6,
            'test': 0.2
        }, use_lengths=False)

        self.assertEqual(self.corpus.num_utterances,
                         sum([sv.num_utterances for sv in res.values()]))

    def test_split_by_proportionally_distribute_labels_by_number_seed(self):
        corpus = resources.create_multi_label_corpus()
        splitter = splitting.Splitter(corpus, random_seed=15)
        res1 = splitter.split_by_proportionally_distribute_labels({
            'train': 0.6,
            'test': 0.2
        }, use_lengths=False)

        corpus = resources.create_multi_label_corpus()
        splitter = splitting.Splitter(corpus, random_seed=15)
        res2 = splitter.split_by_proportionally_distribute_labels({
            'train': 0.6,
            'test': 0.2
        }, use_lengths=False)

        self.assertSetEqual(set(res1['train'].utterances.keys()),
                            set(res2['train'].utterances.keys()))
        self.assertSetEqual(set(res1['test'].utterances.keys()),
                            set(res2['test'].utterances.keys()))
