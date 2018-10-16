from audiomate import corpus
from audiomate.corpus import assets
from audiomate.corpus import validation

from tests import resources


class TestUtteranceTranscriptionRatioValidator:

    def test_name(self):
        val = validation.UtteranceTranscriptionRatioValidator(10, corpus.LL_WORD_TRANSCRIPT)

        assert val.name() == 'Utterance-Transcription-Ratio ({})'.format(corpus.LL_WORD_TRANSCRIPT)

    def test_validate(self):
        ds = resources.create_dataset()
        ds.utterances['utt-3'].label_lists[corpus.LL_WORD_TRANSCRIPT][0].value = 'max length here 11'
        ds.utterances['utt-4'].label_lists[corpus.LL_WORD_TRANSCRIPT][0].value = 'too long here'

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
        ds.utterances['utt-3'].label_lists[corpus.LL_WORD_TRANSCRIPT].labels = []
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

    def test_validate(self):
        ds = resources.create_single_label_corpus()
        utt4_ll = assets.LabelList(idx='default', labels=[
            assets.Label('a', start=0.0, end=1.44),
            assets.Label('a', start=1.89, end=10.0),
        ])
        ds.utterances['utt-4'].set_label_list(utt4_ll)
        utt6_ll = assets.LabelList(idx='default', labels=[
            assets.Label('a', start=1.33, end=5.9),
            assets.Label('a', start=5.9, end=14.7),
        ])
        ds.utterances['utt-6'].set_label_list(utt6_ll)

        val = validation.LabelCoverageValidator('default')
        result = val.validate(ds)

        assert not result.passed
        assert set(result.uncovered_segments.keys()) == {'utt-4', 'utt-6'}

        assert result.uncovered_segments['utt-4'] == [(1.44, 1.89)]
        assert result.uncovered_segments['utt-6'] == [(0.0, 1.33), (14.7, 15.0)]

    def test_validate_passes(self):
        ds = resources.create_single_label_corpus()

        val = validation.LabelCoverageValidator('default')
        result = val.validate(ds)

        assert result.passed
        assert len(result.uncovered_segments) == 0
