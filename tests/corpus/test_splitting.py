import unittest

from pingu.corpus import splitting

from .. import resources


class SplitterTest(unittest.TestCase):
    def setUp(self):
        self.corpus = resources.create_dataset()
        self.splitter = splitting.Splitter(self.corpus)

    def test_split_by_number_of_utterances(self):
        res = self.splitter.split_by_number_of_utterances({
            'train' : 0.6,
            'test' : 0.4
        })

        self.assertEqual(3, res['train'].num_utterances)
        self.assertEqual(2, res['test'].num_utterances)

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
        res = self.splitter.get_identifiers_randomly_splitted(identifiers=[
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'v', 't'
        ], proportions={
            'a': 0.3333,
            'b': 0.6666
        })

        self.assertEqual(4, len(res['a']))
        self.assertEqual(8, len(res['b']))
        self.assertEqual(12, len(set(res['a'] + res['b'])))
