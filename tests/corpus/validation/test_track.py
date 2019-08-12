import audiomate
from audiomate.corpus import validation

from tests import resources


class TestTrackReadValidator:

    def test_name(self):
        val = validation.TrackReadValidator()
        assert val.name() == 'Track-Read'

    def test_validate_passes(self):
        corpus = audiomate.Corpus()
        corpus.new_file(
            resources.sample_wav_file('wav_1.wav'),
            'wav1'
        )
        corpus.new_file(
            resources.sample_wav_file('wav_2.wav'),
            'wav2'
        )

        val = validation.TrackReadValidator()
        res = val.validate(corpus)

        assert res.passed

    def test_validate_doesnt_pass(self):
        corpus = audiomate.Corpus()
        corpus.new_file(
            resources.sample_wav_file('wav_1.wav'),
            'wav1'
        )
        corpus.new_file(
            resources.sample_wav_file('invalid_audio.wav'),
            'wav2'
        )

        val = validation.TrackReadValidator()
        res = val.validate(corpus)

        assert not res.passed
        assert len(res.invalid_items) == 1
        assert 'wav2' in res.invalid_items
