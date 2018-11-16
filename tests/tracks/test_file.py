import os

import pytest
import numpy as np
import librosa

from audiomate import tracks

from tests import resources


@pytest.fixture()
def audio_path():
    return os.path.join(os.path.dirname(resources.__file__), 'audio_formats')


class TestFile:

    @pytest.mark.parametrize('name,sampling_rate', [
        ('flac_1_16k_16b.flac', 16000),
        ('mp3_2_44_1k_16b.mp3', 44100),
        ('wav_1_16k_24b.wav', 16000),
        ('wav_2_44_1k_16b.wav', 44100),
        ('wavex_2_48k_24b.wav', 48000)
    ])
    def test_sampling_rate(self, name, sampling_rate, audio_path):
        file_obj = tracks.FileTrack('some_idx', os.path.join(audio_path, name))

        assert file_obj.sampling_rate == sampling_rate

    @pytest.mark.parametrize('name,num_channels', [
        ('flac_1_16k_16b.flac', 1),
        ('mp3_2_44_1k_16b.mp3', 2),
        ('wav_1_16k_24b.wav', 1),
        ('wav_2_44_1k_16b.wav', 2),
        ('wavex_2_48k_24b.wav', 2)
    ])
    def test_num_channels(self, name, num_channels, audio_path):
        file_obj = tracks.FileTrack('some_idx', os.path.join(audio_path, name))

        assert file_obj.num_channels == num_channels

    @pytest.mark.parametrize('name,num_samples', [
        ('wav_1_16k_24b.wav', 41523),
        ('wav_2_44_1k_16b.wav', 176400),
        ('wavex_2_48k_24b.wav', 192000)
    ])
    def test_num_samples(self, name, num_samples, audio_path):
        file_obj = tracks.FileTrack('some_idx', os.path.join(audio_path, name))

        assert file_obj.num_samples == num_samples

    @pytest.mark.parametrize('name,num_samples', [
        ('flac_1_16k_16b.flac', 103424),
        ('mp3_2_44_1k_16b.mp3', 222336)
    ])
    def test_num_samples_compressed_formats(self, name, num_samples, audio_path):
        file_obj = tracks.FileTrack('some_idx', os.path.join(audio_path, name))

        assert file_obj.num_samples == pytest.approx(num_samples, abs=2000)

    @pytest.mark.parametrize('name,duration', [
        ('wav_1_16k_24b.wav', 2.5951875),
        ('wav_2_44_1k_16b.wav', 4.0),
        ('wavex_2_48k_24b.wav', 4.0)
    ])
    def test_num_duration(self, name, duration, audio_path):
        file_obj = tracks.FileTrack('some_idx', os.path.join(audio_path, name))

        assert file_obj.duration == pytest.approx(duration)

    @pytest.mark.parametrize('name,duration', [
        ('flac_1_16k_16b.flac', 6.464),
        ('mp3_2_44_1k_16b.mp3', 5.0416326531)
    ])
    def test_num_duration_compressed_formats(self, name, duration, audio_path):
        file_obj = tracks.FileTrack('some_idx', os.path.join(audio_path, name))

        assert file_obj.duration == pytest.approx(duration, abs=0.1)

    @pytest.mark.parametrize('name', [
        ('flac_1_16k_16b.flac'),
        ('mp3_2_44_1k_16b.mp3'),
        ('wav_1_16k_24b.wav'),
        ('wav_2_44_1k_16b.wav'),
        ('wavex_2_48k_24b.wav')
    ])
    def test_read_samples(self, name, audio_path):
        audio_path = os.path.join(audio_path, name)
        file_obj = tracks.FileTrack('some_idx', audio_path)

        expected, __ = librosa.core.load(audio_path, sr=None, mono=True)
        actual = file_obj.read_samples()

        assert np.array_equal(actual, expected)

    @pytest.mark.parametrize('name', [
        ('flac_1_16k_16b.flac'),
        ('mp3_2_44_1k_16b.mp3'),
        ('wav_1_16k_24b.wav'),
        ('wav_2_44_1k_16b.wav'),
        ('wavex_2_48k_24b.wav')
    ])
    def test_read_samples_fix_sampling_rate(self, name, audio_path):
        audio_path = os.path.join(audio_path, name)
        file_obj = tracks.FileTrack('some_idx', audio_path)

        expected, __ = librosa.core.load(audio_path, sr=16000, mono=True)
        actual = file_obj.read_samples(sr=16000)

        assert np.array_equal(actual, expected)

    @pytest.mark.parametrize('name', [
        ('flac_1_16k_16b.flac'),
        ('mp3_2_44_1k_16b.mp3'),
        ('wav_1_16k_24b.wav'),
        ('wav_2_44_1k_16b.wav'),
        ('wavex_2_48k_24b.wav')
    ])
    def test_read_samples_range(self, name, audio_path):
        audio_path = os.path.join(audio_path, name)
        file_obj = tracks.FileTrack('some_idx', audio_path)

        expected, __ = librosa.core.load(audio_path, sr=None, mono=True,
                                         offset=1.0, duration=1.7)
        actual = file_obj.read_samples(offset=1.0, duration=1.7)

        assert np.array_equal(actual, expected)

    def test_read_frames(self, tmpdir):
        wav_path = os.path.join(tmpdir.strpath, 'file.wav')
        wav_content = np.random.random(10044)

        librosa.output.write_wav(wav_path, wav_content, 16000)
        file_obj = tracks.FileTrack('some_idx', wav_path)

        data = list(file_obj.read_frames(frame_size=400, hop_size=160))
        frames = np.array([x[0] for x in data])
        last = [x[1] for x in data]

        assert frames.shape == (62, 400)
        assert frames.dtype == np.float32
        assert np.allclose(frames[0], wav_content[:400], atol=0.0001)
        expect = np.pad(wav_content[9760:], (0, 116), mode='constant')
        assert np.allclose(frames[61], expect, atol=0.0001)

        assert last[:-1] == [False] * (len(data) - 1)
        assert last[-1]
