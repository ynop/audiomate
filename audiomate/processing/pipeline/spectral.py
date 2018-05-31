import numpy as np
import scipy.fftpack as fft
import librosa
from librosa import filters
from librosa import util

from . import base


def stft_from_frames(frames, window='hann', dtype=np.complex64):
    """
    Variation of the librosa.core.stft function,
    that computes the short-time-fourier-transfrom from frames instead from the signal.

    See http://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#stft
    """

    win_length = frames.shape[0]
    n_fft = win_length

    fft_window = filters.get_window(window, win_length, fftbins=True)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), frames.shape[1]),
                           dtype=dtype,
                           order='F')

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(util.MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                          stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        # RFFT and Conjugate here to match phase from DPWE code
        stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window *
                                            frames[:, bl_s:bl_t],
                                            axis=0)[:stft_matrix.shape[0]].conj()

    return stft_matrix


class MelSpectrogram(base.Computation):
    """
    Computation step that extracts mel-spectrogram features from the given frames.

    Based on http://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html

    Args:
        n_mels (int): Number of mel bands to generate.
    """

    def __init__(self, n_mels=128, parent=None, name=None):
        super(MelSpectrogram, self).__init__(parent=parent, name=name)

        self.n_mels = n_mels

    def compute(self, chunk, sampling_rate, corpus=None, utterance=None):
        power_spec = np.abs(stft_from_frames(chunk.data.T)) ** 2
        mel = librosa.feature.melspectrogram(S=power_spec, n_mels=self.n_mels, sr=sampling_rate)

        return mel.T


class MFCC(base.Computation):
    """
    Computation step that extracts mfcc features from the given frames.

    Based on http://librosa.github.io/librosa/generated/librosa.feature.mfcc.html

    Args:
        n_mels (int): Number of mel bands to generate.
        n_mfcc (int): number of MFCCs to return.
    """

    def __init__(self, n_mfcc=13, n_mels=128, parent=None, name=None):
        super(MFCC, self).__init__(parent=parent, name=name)

        self.n_mfcc = n_mfcc
        self.n_mels = n_mels

    def compute(self, chunk, sampling_rate, corpus=None, utterance=None):
        power_spec = np.abs(stft_from_frames(chunk.data.T)) ** 2

        mel = librosa.feature.melspectrogram(S=power_spec, n_mels=self.n_mels, sr=sampling_rate)
        mel_power = librosa.power_to_db(mel)
        mfcc = librosa.feature.mfcc(S=mel_power, n_mfcc=self.n_mfcc)

        return mfcc.T
