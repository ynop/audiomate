import librosa
import numpy as np

from . import base
from . import spectral


class OnsetStrength(base.Computation):
    """
    Compute a spectral flux onset strength envelope.

    Based on http://librosa.github.io/librosa/generated/librosa.onset.onset_strength.html

    Args:
        n_mels (int): Number of mel bands to generate.
    """

    def __init__(self, n_mels=128, parent=None, name=None):
        super(OnsetStrength, self).__init__(left_context=1, right_context=0, parent=parent, name=name)

        self.n_mels = n_mels

    def compute(self, chunk, sampling_rate, corpus=None, utterance=None):
        # Compute mel-spetrogram
        power_spec = np.abs(spectral.stft_from_frames(chunk.data.T)) ** 2
        mel = np.abs(librosa.feature.melspectrogram(S=power_spec, n_mels=self.n_mels, sr=sampling_rate))
        mel_power = librosa.power_to_db(mel)

        # Compute onset strengths
        oenv = librosa.onset.onset_strength(S=mel_power, center=False)

        # Switch dimensions and add dimension to have frames
        oenv = oenv.T.reshape(oenv.shape[0], -1)

        # Remove context
        oenv = oenv[chunk.left_context:oenv.shape[0] - chunk.right_context]

        return oenv
