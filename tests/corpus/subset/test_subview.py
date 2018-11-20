from audiomate import annotations
from audiomate import tracks
from audiomate.corpus.subset import subview

import pytest

from tests import resources


class TestMatchingUtteranceIdxFilter:

    def test_match(self):
        filter = subview.MatchingUtteranceIdxFilter(utterance_idxs={'a', 'b', 'd'})

        assert filter.match(tracks.Utterance('a', 'x'), None)
        assert filter.match(tracks.Utterance('b', 'x'), None)
        assert filter.match(tracks.Utterance('d', 'x'), None)
        assert not filter.match(tracks.Utterance('c', 'x'), None)
        assert not filter.match(tracks.Utterance('e', 'x'), None)

    def test_match_inverse(self):
        filter = subview.MatchingUtteranceIdxFilter(utterance_idxs={'a', 'b', 'd'}, inverse=True)

        assert not filter.match(tracks.Utterance('a', 'x'), None)
        assert not filter.match(tracks.Utterance('b', 'x'), None)
        assert not filter.match(tracks.Utterance('d', 'x'), None)
        assert filter.match(tracks.Utterance('c', 'x'), None)
        assert filter.match(tracks.Utterance('e', 'x'), None)

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


@pytest.fixture
def utt_without_noise():
    utt = tracks.Utterance('utt-1', 'file-1')

    utt.set_label_list(annotations.LabelList(idx='alpha', labels=[
        annotations.Label('music', 0, 5),
        annotations.Label('speech', 5, 12),
        annotations.Label('music', 13, 15)
    ]))

    utt.set_label_list(annotations.LabelList(idx='bravo', labels=[
        annotations.Label('music', 0, 1),
        annotations.Label('speech', 2, 6)
    ]))

    return utt


@pytest.fixture
def utt_with_noise():
    utt = tracks.Utterance('utt-2', 'file-2')

    utt.set_label_list(annotations.LabelList(idx='alpha', labels=[
        annotations.Label('music', 0, 5),
        annotations.Label('speech', 5, 12),
        annotations.Label('noise', 13, 15)
    ]))

    utt.set_label_list(annotations.LabelList(idx='bravo', labels=[
        annotations.Label('music', 0, 1),
        annotations.Label('speech', 2, 6)
    ]))

    return utt


class TestMatchingLabelFilter:

    def test_match_all_label_lists(self, utt_with_noise, utt_without_noise):
        filter = subview.MatchingLabelFilter(labels={'music', 'speech'})

        assert filter.match(utt_without_noise, None)
        assert not filter.match(utt_with_noise, None)

    def test_match_single(self, utt_with_noise, utt_without_noise):
        filter = subview.MatchingLabelFilter(labels={'music', 'speech'}, label_list_ids={'alpha'})

        assert filter.match(utt_without_noise, None)
        assert not filter.match(utt_with_noise, None)

        filter = subview.MatchingLabelFilter(labels={'music', 'speech'}, label_list_ids={'bravo'})

        assert filter.match(utt_without_noise, None)
        assert filter.match(utt_with_noise, None)

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


@pytest.fixture
def sample_subview():
        filter = subview.MatchingUtteranceIdxFilter(utterance_idxs={'utt-1', 'utt-3'})
        corpus = resources.create_dataset()
        return subview.Subview(corpus, filter_criteria=[filter])


class TestSubview:

    def test_tracks(self, sample_subview):
        assert sample_subview.num_tracks == 2
        assert 'wav-1' in sample_subview.tracks.keys()
        assert 'wav_3' in sample_subview.tracks.keys()

    def test_utterances(self, sample_subview):
        assert sample_subview.num_utterances == 2
        assert 'utt-1' in sample_subview.utterances.keys()
        assert 'utt-3' in sample_subview.utterances.keys()

    def test_issuers(self, sample_subview):
        assert sample_subview.num_issuers == 2
        assert 'spk-1' in sample_subview.issuers.keys()
        assert 'spk-2' in sample_subview.issuers.keys()

    def test_serialize(self, sample_subview):
        repr = sample_subview.serialize()

        assert repr == 'matching_utterance_ids\ninclude,utt-1,utt-3'

    def test_parse(self):
        corpus = resources.create_dataset()
        sv = subview.Subview.parse('matching_utterance_ids\ninclude,utt-1,utt-3', corpus=corpus)

        assert len(sv.filter_criteria) == 1
        assert sv.filter_criteria[0].utterance_idxs == {'utt-1', 'utt-3'}

    def test_utterances_without_issuers(self, sample_subview):
        sample_subview.corpus.utterances['utt-3'].issuer = None
        sample_subview.corpus.utterances['utt-4'].issuer = None
        sample_subview.corpus.utterances['utt-5'].issuer = None

        assert sample_subview.num_utterances == 2
        assert sample_subview.num_issuers == 1
