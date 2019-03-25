import os

from audiomate import corpus
from audiomate.corpus import io
from audiomate import issuers

from tests import resources
from . import reader_test as rt


class TestTimitReader(rt.CorpusReaderTest):

    SAMPLE_PATH = resources.sample_corpus_path('timit')
    FILE_TRACK_BASE_PATH = SAMPLE_PATH

    EXPECTED_NUMBER_OF_TRACKS = 9
    EXPECTED_TRACKS = [
        rt.ExpFileTrack('dr1-mkls0-sa1', os.path.join('TRAIN', 'DR1', 'MKLS0', 'SA1.WAV')),
        rt.ExpFileTrack('dr1-mkls0-sa2', os.path.join('TRAIN', 'DR1', 'MKLS0', 'SA2.WAV')),
        rt.ExpFileTrack('dr1-mrcg0-sx78', os.path.join('TRAIN', 'DR1', 'MRCG0', 'SX78.WAV')),
        rt.ExpFileTrack('dr2-mkjo0-si1517', os.path.join('TRAIN', 'DR2', 'MKJO0', 'SI1517.WAV')),
        rt.ExpFileTrack('dr2-mrfk0-sx176', os.path.join('TRAIN', 'DR2', 'MRFK0', 'SX176.WAV')),
        rt.ExpFileTrack('dr1-fdac1-sa2', os.path.join('TEST', 'DR1', 'FDAC1', 'SA2.WAV')),
        rt.ExpFileTrack('dr1-mjsw0-sa1', os.path.join('TEST', 'DR1', 'MJSW0', 'SA1.WAV')),
        rt.ExpFileTrack('dr1-mjsw0-sx20', os.path.join('TEST', 'DR1', 'MJSW0', 'SX20.WAV')),
        rt.ExpFileTrack('dr2-fpas0-sx224', os.path.join('TEST', 'DR2', 'FPAS0', 'SX224.WAV'))
    ]

    EXPECTED_NUMBER_OF_UTTERANCES = 9
    EXPECTED_UTTERANCES = [
        rt.ExpUtterance('dr1-mkls0-sa1', 'dr1-mkls0-sa1', 'KLS0', 0, float('inf')),
        rt.ExpUtterance('dr1-mkls0-sa2', 'dr1-mkls0-sa2', 'KLS0', 0, float('inf')),
        rt.ExpUtterance('dr1-mrcg0-sx78', 'dr1-mrcg0-sx78', 'RCG0', 0, float('inf')),
        rt.ExpUtterance('dr2-mkjo0-si1517', 'dr2-mkjo0-si1517', 'KJO0', 0, float('inf')),
        rt.ExpUtterance('dr2-mrfk0-sx176', 'dr2-mrfk0-sx176', 'RFK0', 0, float('inf')),
        rt.ExpUtterance('dr1-fdac1-sa2', 'dr1-fdac1-sa2', 'DAC1', 0, float('inf')),
        rt.ExpUtterance('dr1-mjsw0-sa1', 'dr1-mjsw0-sa1', 'JSW0', 0, float('inf')),
        rt.ExpUtterance('dr1-mjsw0-sx20', 'dr1-mjsw0-sx20', 'JSW0', 0, float('inf')),
        rt.ExpUtterance('dr2-fpas0-sx224', 'dr2-fpas0-sx224', 'PAS0', 0, float('inf'))
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 7
    EXPECTED_ISSUERS = [
        rt.ExpSpeaker('DAC1', 1, issuers.Gender.FEMALE, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('JSW0', 2, issuers.Gender.MALE, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('PAS0', 1, issuers.Gender.FEMALE, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('KLS0', 2, issuers.Gender.MALE, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('RCG0', 1, issuers.Gender.MALE, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('KJO0', 1, issuers.Gender.MALE, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('RFK0', 1, issuers.Gender.MALE, issuers.AgeGroup.UNKNOWN, None)
    ]

    RAW_LL = corpus.LL_WORD_TRANSCRIPT_RAW
    WORD_LL = corpus.LL_WORD_TRANSCRIPT
    PHONE_LL = corpus.LL_PHONE_TRANSCRIPT

    EXPECTED_LABEL_LISTS = {
        'dr1-mkls0-sa1': [
            rt.ExpLabelList(RAW_LL, 1),
            rt.ExpLabelList(WORD_LL, 11),
            rt.ExpLabelList(PHONE_LL, 42),
        ],
        'dr1-mkls0-sa2': [
            rt.ExpLabelList(RAW_LL, 1),
            rt.ExpLabelList(WORD_LL, 10),
            rt.ExpLabelList(PHONE_LL, 33),
        ],
        'dr1-mrcg0-sx78': [
            rt.ExpLabelList(RAW_LL, 1),
            rt.ExpLabelList(WORD_LL, 5),
            rt.ExpLabelList(PHONE_LL, 31),
        ],
        'dr2-mkjo0-si1517': [
            rt.ExpLabelList(RAW_LL, 1),
            rt.ExpLabelList(WORD_LL, 7),
            rt.ExpLabelList(PHONE_LL, 42),
        ],
        'dr2-mrfk0-sx176': [
            rt.ExpLabelList(RAW_LL, 1),
        ],
        'dr1-fdac1-sa2': [
            rt.ExpLabelList(RAW_LL, 1),
        ],
        'dr1-mjsw0-sa1': [
            rt.ExpLabelList(RAW_LL, 1),
        ],
        'dr1-mjsw0-sx20': [
            rt.ExpLabelList(RAW_LL, 1),
        ],
        'dr2-fpas0-sx224': [
            rt.ExpLabelList(RAW_LL, 1),
        ],
    }

    EXPECTED_LABELS = {
        'dr1-mkls0-sa1': [
            rt.ExpLabel(RAW_LL, 'She had your dark suit in greasy wash water all year.', 0, float('inf')),
            rt.ExpLabel(WORD_LL, 'she', 0.210625, 0.4275),
            rt.ExpLabel(PHONE_LL, 'h#', 0.0, 0.210625),
        ],
        'dr1-mkls0-sa2': [
            rt.ExpLabel(RAW_LL, 'Don\'t ask me to carry an oily rag like that.', 0, float('inf')),
            rt.ExpLabel(WORD_LL, 'ask', 0.3625, 0.645),
            rt.ExpLabel(PHONE_LL, 'd', 0.209375, 0.244375),
        ],
        'dr1-mrcg0-sx78': [
            rt.ExpLabel(RAW_LL, 'Doctors prescribe drugs too freely.', 0, float('inf')),
            rt.ExpLabel(WORD_LL, 'freely', 1.8575, 2.2898125),
            rt.ExpLabel(PHONE_LL, 't', 0.348125, 0.37375),
        ],
        'dr2-mkjo0-si1517': [
            rt.ExpLabel(RAW_LL, 'Hired, hard lackeys of the warmongering capitalists.', 0, float('inf')),
            rt.ExpLabel(WORD_LL, 'warmongering', 1.461125, 2.18275),
            rt.ExpLabel(PHONE_LL, 'h#', 2.9075, 3.1),
        ],
        'dr2-mrfk0-sx176': [
            rt.ExpLabel(RAW_LL, 'Buying a thoroughbred horse requires intuition and expertise.', 0, float('inf'))
        ],
        'dr1-fdac1-sa2': [
            rt.ExpLabel(RAW_LL, 'Don\'t ask me to carry an oily rag like that.', 0, float('inf')),
        ],
        'dr1-mjsw0-sa1': [
            rt.ExpLabel(RAW_LL, 'She had your dark suit in greasy wash water all year.', 0, float('inf')),
        ],
        'dr1-mjsw0-sx20': [
            rt.ExpLabel(RAW_LL, 'She wore warm, fleecy, woolen overalls.', 0, float('inf')),
        ],
        'dr2-fpas0-sx224': [
            rt.ExpLabel(RAW_LL, 'How good is your endurance?', 0, float('inf')),
        ],
    }

    EXPECTED_NUMBER_OF_SUBVIEWS = 2
    EXPECTED_SUBVIEWS = [
        rt.ExpSubview('TRAIN', [
            'dr1-mkls0-sa1',
            'dr1-mkls0-sa2',
            'dr1-mrcg0-sx78',
            'dr2-mkjo0-si1517',
            'dr2-mrfk0-sx176',
        ]),
        rt.ExpSubview('TEST', [
            'dr1-fdac1-sa2',
            'dr1-mjsw0-sa1',
            'dr1-mjsw0-sx20',
            'dr2-fpas0-sx224',
        ]),
    ]

    def load(self):
        return io.TimitReader().load(self.SAMPLE_PATH)
