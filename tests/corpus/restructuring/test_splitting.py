import unittest

import pytest

from pingu.corpus.restructuring import splitting
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
        assert train_duration / test_duration == pytest.approx(3, rel=5.0)

    def test_split_by_number_of_utterances(self):
        res = self.splitter.split_by_number_of_utterances({
            'train': 0.6,
            'test': 0.2
        })

        self.assertEqual(6, res['train'].num_utterances)
        self.assertEqual(2, res['test'].num_utterances)

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

    def test_split_by_proportionally_distribute_labels(self):
        res = self.splitter.split_by_proportionally_distribute_labels({
            'train': 0.6,
            'test': 0.2
        })

        self.assertEqual(self.corpus.num_utterances,
                         sum([sv.num_utterances for sv in res.values()]))

    def test_split_by_proportionally_distribute_labels_seed(self):
        corpus = resources.create_multi_label_corpus()
        splitter = splitting.Splitter(corpus, random_seed=15)
        res1 = splitter.split_by_proportionally_distribute_labels({
            'train': 0.6,
            'test': 0.2
        })

        corpus = resources.create_multi_label_corpus()
        splitter = splitting.Splitter(corpus, random_seed=15)
        res2 = splitter.split_by_proportionally_distribute_labels({
            'train': 0.6,
            'test': 0.2
        })

        self.assertSetEqual(set(res1['train'].utterances.keys()),
                            set(res2['train'].utterances.keys()))
        self.assertSetEqual(set(res1['test'].utterances.keys()),
                            set(res2['test'].utterances.keys()))

    def test_absolute_proportions(self):
        res = self.splitter.absolute_proportions({
            'a': 0.6,
            'b': 0.2,
            'c': 0.2
        }, 120)

        self.assertEqual(72, res['a'])
        self.assertEqual(24, res['b'])
        self.assertEqual(24, res['c'])

    def test_get_identifiers_randomly_splitted(self):
        res = self.splitter.split_identifiers(identifiers=[
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'v', 't'
        ], proportions={
            'a': 0.3333,
            'b': 0.6666
        })

        self.assertEqual(4, len(res['a']))
        self.assertEqual(8, len(res['b']))
        self.assertEqual(12, len(set(res['a'] + res['b'])))

    def test_get_identifiers_splitted_by_weights_single_category(self):
        identifiers = {
            'a': {'mi': 3},
            'b': {'mi': 4},
            'c': {'mi': 6},
            'd': {'mi': 1},
            'e': {'mi': 4},
            'f': {'mi': 5},
            'g': {'mi': 3}
        }

        proportions = {
            'train': 0.5,
            'test': 0.25,
            'dev': 0.25
        }

        res = splitting.Splitter.get_identifiers_splitted_by_weights(identifiers=identifiers,
                                                                     proportions=proportions)

        self.assertGreater(len(res['train']), 0)
        self.assertGreater(len(res['test']), 0)
        self.assertGreater(len(res['dev']), 0)
        self.assertEqual(len(identifiers), sum([len(x) for x in res.values()]))

    def test_get_identifiers_splitted_by_weights(self):
        identifiers = {
            'a': {'mi': 3, 'ma': 2, 'mu': 1},
            'b': {'mi': 4, 'ma': 5, 'mu': 4},
            'c': {'mi': 6, 'ma': 4, 'mu': 3},
            'd': {'mi': 1, 'ma': 3, 'mu': 2},
            'e': {'mi': 4, 'ma': 1, 'mu': 5},
            'f': {'mi': 5, 'ma': 4, 'mu': 3},
            'g': {'mi': 3, 'ma': 4, 'mu': 5}
        }

        proportions = {
            'train': 0.5,
            'test': 0.25,
            'dev': 0.25
        }

        res = splitting.Splitter.get_identifiers_splitted_by_weights(identifiers=identifiers,
                                                                     proportions=proportions)

        self.assertGreater(len(res['train']), 0)
        self.assertGreater(len(res['test']), 0)
        self.assertGreater(len(res['dev']), 0)
        self.assertEqual(len(identifiers), sum([len(x) for x in res.values()]))
