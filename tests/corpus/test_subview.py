import unittest

from pingu.corpus import assets
from pingu.corpus import subview

from .. import resources


class MatchingUtteranceIdxFilterTest(unittest.TestCase):
    def test_match(self):
        filter = subview.MatchingUtteranceIdxFilter(utterance_idxs={'a', 'b', 'd'})

        self.assertTrue(filter.match(assets.Utterance('a', 'x'), None))
        self.assertTrue(filter.match(assets.Utterance('b', 'x'), None))
        self.assertTrue(filter.match(assets.Utterance('d', 'x'), None))
        self.assertFalse(filter.match(assets.Utterance('c', 'x'), None))
        self.assertFalse(filter.match(assets.Utterance('e', 'x'), None))

    def test_match_inverse(self):
        filter = subview.MatchingUtteranceIdxFilter(utterance_idxs={'a', 'b', 'd'}, inverse=True)

        self.assertFalse(filter.match(assets.Utterance('a', 'x'), None))
        self.assertFalse(filter.match(assets.Utterance('b', 'x'), None))
        self.assertFalse(filter.match(assets.Utterance('d', 'x'), None))
        self.assertTrue(filter.match(assets.Utterance('c', 'x'), None))
        self.assertTrue(filter.match(assets.Utterance('e', 'x'), None))

    def test_serialize(self):
        f = subview.MatchingUtteranceIdxFilter(utterance_idxs={'a', 'b', 'd'})
        assert f.serialize() == 'include,a,b,d'

    def test_serialize_inverse(self):
        f = subview.MatchingUtteranceIdxFilter(utterance_idxs={'a', 'b', 'd'}, inverse=True)
        assert f.serialize() == 'exclude,a,b,d'

    def test_parse(self):
        f = subview.MatchingUtteranceIdxFilter.parse('include,a,b,d')

        assert f.utterance_idxs == {'a', 'b', 'd'}
        assert not f.inverse

    def test_parse_inverse(self):
        f = subview.MatchingUtteranceIdxFilter.parse('exclude,a,b,d')

        assert f.utterance_idxs == {'a', 'b', 'd'}
        assert f.inverse


class SubviewTest(unittest.TestCase):

    def setUp(self):
        filter = subview.MatchingUtteranceIdxFilter(utterance_idxs={'utt-1', 'utt-3'})
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

    def test_serialize(self):
        repr = self.subview.serialize()

        assert repr == 'matching_utterance_ids\ninclude,utt-1,utt-3'

    def test_parse(self):
        sv = subview.Subview.parse('matching_utterance_ids\ninclude,utt-1,utt-3', corpus=self.corpus)

        assert sv.corpus == self.corpus
        assert len(sv.filter_criteria) == 1
        assert sv.filter_criteria[0].utterance_idxs == {'utt-1', 'utt-3'}
