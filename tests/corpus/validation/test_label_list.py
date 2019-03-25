from audiomate import corpus
from audiomate import tracks
from audiomate import annotations
from audiomate.corpus import validation

from tests import resources


class TestUtteranceTranscriptionRatioValidator:

    def test_name(self):
        val = validation.UtteranceTranscriptionRatioValidator(10, corpus.LL_WORD_TRANSCRIPT)

        assert val.name() == 'Utterance-Transcription-Ratio ({})'.format(corpus.LL_WORD_TRANSCRIPT)

    def test_validate(self):
        ds = resources.create_dataset()
        ds.utterances['utt-3'].set_label_list(annotations.LabelList.create_single(
            'max length here 11',
            idx=corpus.LL_WORD_TRANSCRIPT
        ))

        ds.utterances['utt-4'].set_label_list(annotations.LabelList.create_single(
            'too long here',
            idx=corpus.LL_WORD_TRANSCRIPT
        ))

        val = validation.UtteranceTranscriptionRatioValidator(10, corpus.LL_WORD_TRANSCRIPT)
        result = val.validate(ds)

        assert not result.passed
        assert len(result.invalid_utterances) == 1
        assert 'utt-4' in result.invalid_utterances.keys()


class TestLabelCountValidator:

    def test_name(self):
        val = validation.LabelCountValidator(1, corpus.LL_WORD_TRANSCRIPT)

        assert val.name() == 'Label-Count ({})'.format(corpus.LL_WORD_TRANSCRIPT)

    def test_validate(self):
        ds = resources.create_dataset()
        ds.utterances['utt-3'].set_label_list(annotations.LabelList(idx=corpus.LL_WORD_TRANSCRIPT))
        del ds.utterances['utt-4'].label_lists[corpus.LL_WORD_TRANSCRIPT]

        val = validation.LabelCountValidator(1, corpus.LL_WORD_TRANSCRIPT)
        result = val.validate(ds)

        assert not result.passed
        assert len(result.invalid_utterances) == 2
        assert result.invalid_utterances['utt-3'] == 'Only {} labels'.format(0)
        assert result.invalid_utterances['utt-4'] == 'No label-list {}'.format(corpus.LL_WORD_TRANSCRIPT)


class TestLabelCoverageValidator:

    def test_name(self):
        val = validation.LabelCoverageValidator('test_ll')

        assert val.name() == 'Label-Coverage (test_ll)'

    def test_validate_passes(self):
        ds = resources.create_single_label_corpus()

        val = validation.LabelCoverageValidator('default')
        result = val.validate(ds)

        assert result.passed
        assert len(result.uncovered_segments) == 0

    def test_validate(self):
        ds = resources.create_single_label_corpus()
        utt4_ll = annotations.LabelList(idx='default', labels=[
            annotations.Label('a', start=0.0, end=1.44),
            annotations.Label('a', start=1.89, end=10.0),
        ])
        ds.utterances['utt-4'].set_label_list(utt4_ll)
        utt6_ll = annotations.LabelList(idx='default', labels=[
            annotations.Label('a', start=1.33, end=5.9),
            annotations.Label('a', start=5.9, end=14.7),
        ])
        ds.utterances['utt-6'].set_label_list(utt6_ll)

        val = validation.LabelCoverageValidator('default')
        result = val.validate(ds)

        assert not result.passed
        assert set(result.uncovered_segments.keys()) == {'utt-4', 'utt-6'}

        assert result.uncovered_segments['utt-4'] == [(1.44, 1.89)]
        assert result.uncovered_segments['utt-6'] == [(0.0, 1.33), (14.7, 15.0)]


class TestLabelOverflowValidator:

    def test_name(self):
        val = validation.LabelOverflowValidator('test_ll')

        assert val.name() == 'Label-Overflow (test_ll)'

    def test_validate_passes(self):
        ds = resources.create_single_label_corpus()

        val = validation.LabelOverflowValidator('default')
        result = val.validate(ds)

        assert result.passed
        assert len(result.overflow_segments) == 0

    def test_validate_returns_part_of_overlapping_label(self):
        ds = resources.create_single_label_corpus()
        utt4_ll = annotations.LabelList(idx='default', labels=[
            annotations.Label('a', start=0.0, end=9.0),
            annotations.Label('b', start=9.0, end=13.0),
        ])
        ds.utterances['utt-4'].set_label_list(utt4_ll)
        utt6_ll = annotations.LabelList(idx='default', labels=[
            annotations.Label('a', start=-2.0, end=5.9),
            annotations.Label('b', start=5.9, end=14.7),
        ])
        ds.utterances['utt-6'].set_label_list(utt6_ll)

        val = validation.LabelOverflowValidator('default')
        result = val.validate(ds)

        assert not result.passed
        assert set(result.overflow_segments.keys()) == {'utt-4', 'utt-6'}

        assert result.overflow_segments['utt-4'] == [(10.0, 13.0, 'b')]
        assert result.overflow_segments['utt-6'] == [(-2.0, 0.0, 'a')]

    def test_validate_utterance_returns_completly_outlying_label(self):
        utt = tracks.Utterance('utt-idx', None, start=10.0, end=17.9)
        ll = annotations.LabelList(idx='default', labels=[
            annotations.Label('a', start=-4.0, end=-2.0),
            annotations.Label('b', start=19.0, end=22.0),
        ])
        utt.set_label_list(ll)

        val = validation.LabelOverflowValidator('default')
        result = val.validate_utterance(utt)
        result = sorted(result, key=lambda x: x[0])

        assert len(result) == 2

        assert result[0] == (-4.0, -2.0, 'a')
        assert result[1] == (19.0, 22.0, 'b')
