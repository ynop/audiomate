from audiomate import corpus
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
