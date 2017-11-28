import unittest

from pingu.corpus import assets
from pingu.corpus import subview

from .. import resources


class MatchingUtteranceIdxFilterTest(unittest.TestCase):
    def test_match(self):
        filter = subview.MatchingUtteranceIdxFilter(utterance_idxs=(['a', 'b', 'd']))

        self.assertTrue(filter.match(assets.Utterance('a', 'x'), None))
        self.assertTrue(filter.match(assets.Utterance('b', 'x'), None))
        self.assertTrue(filter.match(assets.Utterance('d', 'x'), None))
        self.assertFalse(filter.match(assets.Utterance('c', 'x'), None))
        self.assertFalse(filter.match(assets.Utterance('e', 'x'), None))

    def test_match_inverse(self):
        filter = subview.MatchingUtteranceIdxFilter(utterance_idxs=(['a', 'b', 'd']), inverse=True)

        self.assertFalse(filter.match(assets.Utterance('a', 'x'), None))
        self.assertFalse(filter.match(assets.Utterance('b', 'x'), None))
        self.assertFalse(filter.match(assets.Utterance('d', 'x'), None))
        self.assertTrue(filter.match(assets.Utterance('c', 'x'), None))
        self.assertTrue(filter.match(assets.Utterance('e', 'x'), None))


class SubviewTest(unittest.TestCase):

    def setUp(self):
        filter = subview.MatchingUtteranceIdxFilter(utterance_idxs=(['utt-1', 'utt-3']))
        self.corpus = resources.create_dataset()
        self.subview = subview.Subview(self.corpus, filter_criteria=[filter])

    def test_files(self):
        self.assertEqual(2, self.subview.num_files)
        self.assertIn('wav-1', self.subview.files.keys())
        self.assertIn('wav_3', self.subview.files.keys())

    def test_utterances(self):
        self.assertEqual(2, self.subview.num_utterances)
        self.assertIn('utt-1', self.subview.utterances.keys())
        self.assertIn('utt-3', self.subview.utterances.keys())

    def test_issuers(self):
        self.assertEqual(2, self.subview.num_issuers)
        self.assertIn('spk-1', self.subview.issuers.keys())
        self.assertIn('spk-2', self.subview.issuers.keys())

    def test_label_lists(self):
        self.assertEqual(1, len(self.subview.label_lists))
        self.assertEqual(2, len(self.subview.label_lists['default']))
        self.assertIn('utt-1', self.subview.label_lists['default'].keys())
        self.assertIn('utt-3', self.subview.label_lists['default'].keys())
