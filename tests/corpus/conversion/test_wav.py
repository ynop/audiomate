import librosa
import numpy as np

from tests import resources

from audiomate import tracks
from audiomate.corpus import conversion


class TestWavAudioFileConverter:

    def test_does_utt_match_target_format_returns_true(self):
        file_path = resources.sample_wav_file('wav_1.wav')
        track = tracks.FileTrack('t', file_path)
        utt = tracks.Utterance('u', track)

        c = conversion.WavAudioFileConverter()
        assert c._does_utt_match_target_format(utt)

    def test_does_utt_match_target_format_with_invalid_sr_returns_false(self):
        file_path = resources.get_resource_path(('audio_formats', 'wav_2_44_1k_16b.wav'))
        track = tracks.FileTrack('t', file_path)
        utt = tracks.Utterance('u', track)

        c = conversion.WavAudioFileConverter()
        assert not c._does_utt_match_target_format(utt)

    def test_does_utt_match_target_format_with_invalid_format_returns_false(self):
        file_path = resources.get_resource_path(('audio_formats', 'mp3_2_44_1k_16b.mp3'))
        track = tracks.FileTrack('t', file_path)
        utt = tracks.Utterance('u', track)

        c = conversion.WavAudioFileConverter()
        assert not c._does_utt_match_target_format(utt)

    def test_convert_files(self, tmp_path):
        source_path = resources.sample_wav_file('wav_1.wav')
        target_path = tmp_path / 'out.wav'

        files = [(source_path, 0, float('inf'), str(target_path))]

        c = conversion.WavAudioFileConverter()
        c._convert_files(files)

        samples, sr = librosa.core.load(source_path, sr=None)

        stored_samples, stored_sr = librosa.core.load(str(target_path), sr=None)

        assert target_path.is_file()
        assert stored_sr == sr
        assert np.array_equal(stored_samples, samples)

    def test_store_samples_sr_24(self, tmp_path):
        source_path = resources.sample_wav_file('wav_1.wav')
        target_path = tmp_path / 'out.wav'

        files = [(source_path, 0, float('inf'), str(target_path))]

        c = conversion.WavAudioFileConverter(sampling_rate=24000)
        c._convert_files(files)

        samples, sr = librosa.core.load(source_path, sr=24000)

        stored_samples, stored_sr = librosa.core.load(str(target_path), sr=None)

        assert target_path.is_file()
        assert stored_sr == sr
        # Don't take it too exactly
        # With sox 14.4.1 it isn't that precise, expecially the first sample
        assert np.allclose(stored_samples[1:], samples[1:], atol=0.001)
