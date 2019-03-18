import os

from audiomate import corpus
from audiomate import issuers
from audiomate.corpus.io import audio_mnist

from tests import resources
from . import reader_test as rt


class TestAudioMNISTReader(rt.CorpusReaderTest):

    SAMPLE_PATH = resources.sample_corpus_path('audio_mnist')
    FILE_TRACK_BASE_PATH = os.path.join(SAMPLE_PATH, 'data')

    EXPECTED_NUMBER_OF_TRACKS = 7
    EXPECTED_TRACKS = [
        rt.ExpFileTrack('0_01_0', '01/0_01_0.wav'),
        rt.ExpFileTrack('0_01_1', '01/0_01_1.wav'),
        rt.ExpFileTrack('1_01_0', '01/1_01_0.wav'),
        rt.ExpFileTrack('0_02_0', '02/0_02_0.wav'),
        rt.ExpFileTrack('4_03_0', '03/4_03_0.wav'),
        rt.ExpFileTrack('4_03_1', '03/4_03_1.wav'),
        rt.ExpFileTrack('0_04_0', '04/0_04_0.wav'),
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 4
    EXPECTED_ISSUERS = [
        rt.ExpSpeaker('01', 3, issuers.Gender.MALE,
                      issuers.AgeGroup.ADULT, None),
        rt.ExpSpeaker('02', 1, issuers.Gender.MALE,
                      issuers.AgeGroup.ADULT, None),
        rt.ExpSpeaker('03', 2, issuers.Gender.FEMALE,
                      issuers.AgeGroup.ADULT, None),
        rt.ExpSpeaker('04', 1, issuers.Gender.MALE,
                      issuers.AgeGroup.YOUTH, None),
    ]

    EXPECTED_NUMBER_OF_UTTERANCES = 7
    EXPECTED_UTTERANCES = [
        rt.ExpUtterance('0_01_0', '0_01_0', '01', 0, float('inf')),
        rt.ExpUtterance('0_01_1', '0_01_1', '01', 0, float('inf')),
        rt.ExpUtterance('1_01_0', '1_01_0', '01', 0, float('inf')),
        rt.ExpUtterance('0_02_0', '0_02_0', '02', 0, float('inf')),
        rt.ExpUtterance('4_03_0', '4_03_0', '03', 0, float('inf')),
        rt.ExpUtterance('4_03_1', '4_03_1', '03', 0, float('inf')),
        rt.ExpUtterance('0_04_0', '0_04_0', '04', 0, float('inf')),
    ]

    EXPECTED_LABEL_LISTS = {
        '0_01_0': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
        ],
        '1_01_0': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
        ],
        '4_03_0': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
        ],
        '0_04_0': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1)
        ],
    }

    EXPECTED_LABELS = {
        '0_01_0': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, '0', 0, float('inf')),
        ],
        '1_01_0': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, '1', 0, float('inf')),
        ],
        '4_03_0': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, '4', 0, float('inf')),
        ],
        '0_04_0': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, '0', 0, float('inf')),
        ],
    }

    EXPECTED_NUMBER_OF_SUBVIEWS = 0

    def load(self):
        reader = audio_mnist.AudioMNISTReader()
        return reader.load(self.SAMPLE_PATH)
