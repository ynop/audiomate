import numpy as np
import librosa

from audiomate import tracks
from audiomate.processing import pipeline

from tests import resources


class TestTempogram:

    def test_compute(self):
        test_file_path = resources.sample_wav_file('wav_1.wav')
        y, sr = librosa.load(test_file_path, sr=None)
        frames = librosa.util.frame(y, frame_length=2048, hop_length=1024).T

        # EXPECTED
        S = np.abs(librosa.stft(y, center=False, n_fft=2048, hop_length=1024)) ** 2
        S = librosa.feature.melspectrogram(S=S, n_mels=128, sr=sr)
        S = librosa.power_to_db(S)
        onsets = librosa.onset.onset_strength(S=S, center=False)
        exp_tgram = librosa.feature.tempogram(onset_envelope=onsets, sr=sr, win_length=11, center=True).T

        # ACTUAL
        tgram_step = pipeline.Tempogram(win_length=11)
        tgrams = tgram_step.process_frames(frames, sr, last=True)

        assert np.allclose(tgrams, exp_tgram)

    def test_compute_online(self):
        # Data: 41523 samples, 16 kHz
        # yields 40 frames with frame-size 2048 and hop-size 1024
        test_file_path = resources.sample_wav_file('wav_1.wav')
        y, sr = librosa.load(test_file_path, sr=None)

        # EXPECTED
        y_pad = np.pad(y, (0, 1024), mode='constant', constant_values=0)
        S = np.abs(librosa.stft(y_pad, center=False, n_fft=2048, hop_length=1024)) ** 2
        S = librosa.feature.melspectrogram(S=S, n_mels=128, sr=sr)
        S = librosa.power_to_db(S)
        onsets = librosa.onset.onset_strength(S=S, center=False)
        exp_tgram = librosa.feature.tempogram(onset_envelope=onsets, sr=sr, win_length=4, center=True).T

        # ACTUAL
        test_file = tracks.FileTrack('idx', test_file_path)
        tgram_step = pipeline.Tempogram(win_length=4)
        tgram_gen = tgram_step.process_track_online(test_file, 2048, 1024, chunk_size=5)

        chunks = list(tgram_gen)
        tgrams = np.vstack(chunks)

        assert np.allclose(tgrams, exp_tgram)

    def test_compute_cleanup_after_one_utterance(self):
        test_file_path = resources.sample_wav_file('wav_1.wav')
        y, sr = librosa.load(test_file_path, sr=None)
        frames = librosa.util.frame(y, frame_length=2048, hop_length=1024).T

        # EXPECTED
        S = np.abs(librosa.stft(y, center=False, n_fft=2048, hop_length=1024)) ** 2
        S = librosa.feature.melspectrogram(S=S, n_mels=128, sr=sr)
        S = librosa.power_to_db(S)
        onsets = librosa.onset.onset_strength(S=S, center=False)
        exp_tgram = librosa.feature.tempogram(onset_envelope=onsets, sr=sr, win_length=11, center=True).T

        # ACTUAL
        tgram_step = pipeline.Tempogram(win_length=11)

        # FIRST RUN
        tgrams = tgram_step.process_frames(frames, sr, last=True)

        assert np.allclose(tgrams, exp_tgram)

        # SECOND RUN
        tgrams = tgram_step.process_frames(frames, sr, last=True)

        assert np.allclose(tgrams, exp_tgram)
