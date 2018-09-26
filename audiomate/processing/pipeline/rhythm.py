import librosa
import numpy as np

from . import base
from . import spectral


class Tempogram(base.Computation):
    """
    Computation step to compute tempogram

    Based on http://librosa.github.io/librosa/generated/librosa.feature.tempogram.html

    Args:
        n_mels (int): Number of mel bands to generate.
        win_length (int): Length of the onset autocorrelation window (in frames/onset measurements).
                          The default settings (384) corresponds to 384 * hop_length / sr ~= 8.9s.
    """

    def __init__(self, n_mels=128, win_length=384, parent=None, name=None):
        super(Tempogram, self).__init__(min_frames=win_length, left_context=1, right_context=0,
                                        parent=parent, name=name)

        self.n_mels = n_mels
        self.win_length = win_length

        self.rest = None

    def compute(self, chunk, sampling_rate, corpus=None, utterance=None):
        # Cleanup rest if it's the first frame
        if chunk.offset == 0:
            self.rest = None

        # Compute mel-spectrogram
        power_spec = np.abs(spectral.stft_from_frames(chunk.data.T)) ** 2
        mel = np.abs(librosa.feature.melspectrogram(S=power_spec, n_mels=self.n_mels, sr=sampling_rate))
        mel_power = librosa.power_to_db(mel)

        # Compute onset strengths
        oenv = librosa.onset.onset_strength(S=mel_power, center=False)

        # Remove context, otherwise we have duplicate frames while online processing
        oenv = oenv[chunk.left_context:]

        if self.rest is not None:
            all_frames = np.concatenate([self.rest, oenv])
        else:
            # Its the first chunk --> pad to center tempogram windows at the beginning
            all_frames = np.pad(oenv, (self.win_length // 2, 0), mode='linear_ramp', end_values=0)

        if chunk.is_last:
            # Its the last chunk --> pad to center tempogram windows at end
            all_frames = np.pad(all_frames, (0, self.win_length // 2), mode='linear_ramp', end_values=0)

            # Compensate the 1 frame that is too much since we want win-len - 1 additional frames,
            # With an even win-len we would have win-len additional frames
            if self.win_length % 2 == 0:
                all_frames = all_frames[:-1]

        if all_frames.shape[0] >= self.win_length:
            tempogram = librosa.feature.tempogram(onset_envelope=all_frames, sr=sampling_rate,
                                                  win_length=self.win_length, center=False).T

            self.rest = all_frames[tempogram.shape[0]:]

            return tempogram
        else:
            self.rest = all_frames
