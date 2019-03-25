from audiomate.corpus import io

from tests import resources
from . import reader_test as rt


class TestFolderReader(rt.CorpusReaderTest):

    SAMPLE_PATH = resources.sample_corpus_path('folder')
    FILE_TRACK_BASE_PATH = SAMPLE_PATH

    EXPECTED_NUMBER_OF_TRACKS = 7
    EXPECTED_TRACKS = [
        rt.ExpFileTrack('empty', 'empty.wav'),
        rt.ExpFileTrack('wav_1', 'wav_1.wav'),
        rt.ExpFileTrack('wav_2', 'wav_2.wav'),
        rt.ExpFileTrack('wav_3', 'wav_3.wav'),
        rt.ExpFileTrack('wav_4', 'wav_4.wav'),
        rt.ExpFileTrack('wav_200_samples', 'wav_200_samples.wav'),
        rt.ExpFileTrack('wav_invalid', 'wav_invalid.wav'),
    ]

    EXPECTED_NUMBER_OF_UTTERANCES = 7
    EXPECTED_UTTERANCES = [
        rt.ExpUtterance('empty', 'empty', None, 0, float('inf')),
        rt.ExpUtterance('wav_1', 'wav_1', None, 0, float('inf')),
        rt.ExpUtterance('wav_2', 'wav_2', None, 0, float('inf')),
        rt.ExpUtterance('wav_3', 'wav_3', None, 0, float('inf')),
        rt.ExpUtterance('wav_4', 'wav_4', None, 0, float('inf')),
        rt.ExpUtterance('wav_200_samples', 'wav_200_samples', None, 0, float('inf')),
        rt.ExpUtterance('wav_invalid', 'wav_invalid', None, 0, float('inf')),
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 0
    EXPECTED_NUMBER_OF_SUBVIEWS = 0

    def load(self):
        return io.FolderReader().load(self.SAMPLE_PATH)
