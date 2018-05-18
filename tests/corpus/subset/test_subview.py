import unittest

from audiomate.corpus import assets
from audiomate.corpus.subset import subview

from tests import resources


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


class MatchingLabelFilterTest(unittest.TestCase):

    def setUp(self):
        self.utt1 = assets.Utterance('utt-1', 'file-1')

        self.utt1.set_label_list(assets.LabelList(idx='alpha', labels=[
            assets.Label('music', 0, 5),
            assets.Label('speech', 5, 12),
            assets.Label('music', 13, 15)
        ]))

        self.utt1.set_label_list(assets.LabelList(idx='bravo', labels=[
            assets.Label('music', 0, 1),
            assets.Label('speech', 2, 6)
        ]))

        self.utt2 = assets.Utterance('utt-2', 'file-2')

        self.utt2.set_label_list(assets.LabelList(idx='alpha', labels=[
            assets.Label('music', 0, 5),
            assets.Label('speech', 5, 12),
            assets.Label('noise', 13, 15)
        ]))

        self.utt2.set_label_list(assets.LabelList(idx='bravo', labels=[
            assets.Label('music', 0, 1),
            assets.Label('speech', 2, 6)
        ]))

    def test_match_all_label_lists(self):
        filter = subview.MatchingLabelFilter(labels={'music', 'speech'})

        assert filter.match(self.utt1, None)
        assert not filter.match(self.utt2, None)

    def test_match_single(self):
        filter = subview.MatchingLabelFilter(labels={'music', 'speech'}, label_list_ids={'alpha'})

        assert filter.match(self.utt1, None)
        assert not filter.match(self.utt2, None)

        filter = subview.MatchingLabelFilter(labels={'music', 'speech'}, label_list_ids={'bravo'})

        assert filter.match(self.utt1, None)
        assert filter.match(self.utt2, None)

    def test_serialize(self):
        filter = subview.MatchingLabelFilter(labels={'music', 'speech'}, label_list_ids={'alpha'})

        assert filter.serialize() == 'alpha|||music,speech'

    def test_serialize_no_label_list_ids(self):
        filter = subview.MatchingLabelFilter(labels={'music', 'speech'})

        assert filter.serialize() == '|||music,speech'

    def test_parse(self):
        filter = subview.MatchingLabelFilter.parse('alpha|||music,speech')

        assert filter.labels == {'music', 'speech'}
        assert filter.label_list_ids == {'alpha'}

    def test_parse_no_label_list_ids(self):
        filter = subview.MatchingLabelFilter.parse('music,speech')

        assert filter.labels == {'music', 'speech'}
        assert filter.label_list_ids == set()


class SubviewTest(unittest.TestCase):

    def setUp(self):
        filter = subview.MatchingUtteranceIdxFilter(utterance_idxs={'utt-1', 'utt-3'})
        self.corpus = resources.create_dataset()
        self.subview = subview.Subview(self.corpus, filter_criteria=[filter])

    def test_files(self):
        assert self.subview.num_files == 2
        assert 'wav-1' in self.subview.files.keys()
        assert 'wav_3' in self.subview.files.keys()

    def test_utterances(self):
        assert self.subview.num_utterances == 2
        assert 'utt-1' in self.subview.utterances.keys()
        assert 'utt-3' in self.subview.utterances.keys()

    def test_issuers(self):
        assert self.subview.num_issuers == 2
        assert 'spk-1' in self.subview.issuers.keys()
        assert 'spk-2' in self.subview.issuers.keys()

    def test_serialize(self):
        repr = self.subview.serialize()

        assert repr == 'matching_utterance_ids\ninclude,utt-1,utt-3'

    def test_parse(self):
        sv = subview.Subview.parse('matching_utterance_ids\ninclude,utt-1,utt-3', corpus=self.corpus)

        assert sv.corpus == self.corpus
        assert len(sv.filter_criteria) == 1
        assert sv.filter_criteria[0].utterance_idxs == {'utt-1', 'utt-3'}

    def test_utterances_without_issuers(self):
        self.corpus.utterances['utt-3'].issuer = None
        self.corpus.utterances['utt-4'].issuer = None
        self.corpus.utterances['utt-5'].issuer = None

        assert self.subview.num_utterances == 2
        assert self.subview.num_issuers == 1
