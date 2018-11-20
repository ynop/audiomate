import numpy as np
import librosa

from audiomate import tracks
from audiomate.processing import pipeline

from tests import resources


class TestOnsetStrength:

    def test_compute(self):
        test_file_path = resources.sample_wav_file('wav_1.wav')
        y, sr = librosa.load(test_file_path, sr=None)
        frames = librosa.util.frame(y, frame_length=2048, hop_length=1024).T

        # EXPECTED
        S = np.abs(librosa.stft(y, center=False, n_fft=2048, hop_length=1024)) ** 2
        S = librosa.feature.melspectrogram(S=S, n_mels=128, sr=sr)
        S = librosa.power_to_db(S)
        exp_onsets = librosa.onset.onset_strength(S=S, center=False).T
        exp_onsets = exp_onsets.reshape(exp_onsets.shape[0], 1)

        # ACTUAL
        onset = pipeline.OnsetStrength()
        onsets = onset.process_frames(frames, sr, last=True)

        assert np.allclose(onsets, exp_onsets)

    def test_compute_online(self):
        test_file_path = resources.sample_wav_file('wav_1.wav')
        y, sr = librosa.load(test_file_path, sr=None)

        # EXPECTED
        y_pad = np.pad(y, (0, 1024), mode='constant', constant_values=0)
        S = np.abs(librosa.stft(y_pad, center=False, n_fft=2048, hop_length=1024)) ** 2
        S = librosa.feature.melspectrogram(S=S, n_mels=128, sr=sr)
        S = librosa.power_to_db(S)
        exp_onsets = librosa.onset.onset_strength(S=S, center=False).T
        exp_onsets = exp_onsets.reshape(exp_onsets.shape[0], 1)

        # ACTUAL
        test_file = tracks.FileTrack('idx', test_file_path)
        onset = pipeline.OnsetStrength()
        onset_gen = onset.process_track_online(test_file, 2048, 1024, chunk_size=5)

        chunks = list(onset_gen)
        onsets = np.vstack(chunks)

        print(onsets.shape, exp_onsets.shape)

        assert np.allclose(onsets, exp_onsets)
