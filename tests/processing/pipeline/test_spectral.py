import numpy as np
import librosa

from audiomate.processing import pipeline


class TestMelSpectrogram:

    def test_compute(self):
        samples = np.arange(8096).astype(np.float32)
        D = np.abs(librosa.core.stft(samples, n_fft=2048, hop_length=512, center=False)) ** 2
        expected = librosa.feature.melspectrogram(S=D, sr=16000, n_mels=128).T

        frames = librosa.util.frame(samples, frame_length=2048, hop_length=512).T
        mel = pipeline.MelSpectrogram(n_mels=128)
        res = mel.process_frames(frames, sampling_rate=16000)

        assert np.array_equal(expected, res)


class TestMFCC:

    def test_compute(self):
        samples = np.arange(8096).astype(np.float32)
        D = np.abs(librosa.core.stft(samples, n_fft=2048, hop_length=512, center=False)) ** 2
        mel = librosa.feature.melspectrogram(S=D, sr=16000, n_mels=128)
        expected = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=13).T

        frames = librosa.util.frame(samples, frame_length=2048, hop_length=512).T
        mfcc = pipeline.MFCC(n_mfcc=13, n_mels=128)
        res = mfcc.process_frames(frames, sampling_rate=16000)

        assert np.array_equal(expected, res)
