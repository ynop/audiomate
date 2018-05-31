import librosa

from . import base


class PowerToDb(base.Computation):
    """
    Convert a power spectrogram (amplitude squared) to decibel (dB) units.

    See http://librosa.github.io/librosa/generated/librosa.core.power_to_db.html

    Note:

        The output can differ depending on offline or online processing, since it depends on statistics over all values.
        And in online mode it only considers values from a single chunk,
        while in offline mode all values of the whole sequence are considered.
    """

    def __init__(self, ref=1.0, amin=1e-10, top_db=80.0, parent=None, name=None):
        super(PowerToDb, self).__init__(parent=parent, name=name)

        self.ref = ref
        self.amin = amin
        self.top_db = top_db

    def compute(self, chunk, sampling_rate, corpus=None, utterance=None):
        return librosa.power_to_db(chunk.data.T, ref=self.ref, amin=self.amin, top_db=self.top_db).T
