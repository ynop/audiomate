import os

from audiomate import corpus
from audiomate import issuers
from audiomate.corpus.io import common_voice

from tests import resources
from . import reader_test as rt


class TestCommonVoiceReader(rt.CorpusReaderTest):

    SAMPLE_PATH = resources.sample_corpus_path('common_voice')
    FILE_TRACK_BASE_PATH = os.path.join(SAMPLE_PATH, 'clips')

    EXPECTED_NUMBER_OF_TRACKS = 9
    EXPECTED_TRACKS = [
        rt.ExpFileTrack('c4b', 'c4b.mp3'),
        rt.ExpFileTrack('8ea', '8ea.mp3'),
        rt.ExpFileTrack('67c', '67c.mp3'),
        rt.ExpFileTrack('f08', 'f08.mp3'),
        rt.ExpFileTrack('b5c', 'b5c.mp3'),
        rt.ExpFileTrack('8f4', '8f4.mp3'),
        rt.ExpFileTrack('7f4', '7f4.mp3'),
        rt.ExpFileTrack('059', '059.mp3'),
        rt.ExpFileTrack('d08', 'd08.mp3'),
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 7
    EXPECTED_ISSUERS = [
        rt.ExpSpeaker('17e', 1, issuers.Gender.MALE, issuers.AgeGroup.SENIOR, None),
        rt.ExpSpeaker('cb3', 2, issuers.Gender.MALE, issuers.AgeGroup.ADULT, None),
        rt.ExpSpeaker('aa3', 2, issuers.Gender.FEMALE, issuers.AgeGroup.ADULT, None),
        rt.ExpSpeaker('90a', 1, issuers.Gender.MALE, issuers.AgeGroup.ADULT, None),
        rt.ExpSpeaker('b0f', 1, issuers.Gender.UNKNOWN, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('5ec', 1, issuers.Gender.UNKNOWN, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('72d', 1, issuers.Gender.UNKNOWN, issuers.AgeGroup.UNKNOWN, None),
    ]

    EXPECTED_NUMBER_OF_UTTERANCES = 9
    EXPECTED_UTTERANCES = [
        rt.ExpUtterance('c4b', 'c4b', '17e', 0, float('inf')),
        rt.ExpUtterance('8ea', '8ea', 'cb3', 0, float('inf')),
        rt.ExpUtterance('67c', '67c', 'cb3', 0, float('inf')),
        rt.ExpUtterance('f08', 'f08', 'aa3', 0, float('inf')),
        rt.ExpUtterance('b5c', 'b5c', 'aa3', 0, float('inf')),
        rt.ExpUtterance('8f4', '8f4', '90a', 0, float('inf')),
        rt.ExpUtterance('7f4', '7f4', 'b0f', 0, float('inf')),
        rt.ExpUtterance('059', '059', '5ec', 0, float('inf')),
        rt.ExpUtterance('d08', 'd08', '72d', 0, float('inf')),
    ]

    EXPECTED_LABEL_LISTS = {
        'c4b': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
        ],
        'f08': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
        ],
        '8f4': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
        ],
        'd08': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1)
        ],
    }

    EXPECTED_LABELS = {
        'c4b': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'Man sollte', 0, float('inf')),
        ],
        'f08': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'Valentin', 0, float('inf')),
        ],
        '8f4': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'Es', 0, float('inf')),
        ],
        '7f4': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'Zieht euch', 0, float('inf')),
        ],
    }

    EXPECTED_NUMBER_OF_SUBVIEWS = 3

    EXPECTED_SUBVIEWS = [
        rt.ExpSubview('train', [
            'c4b',
            '8ea',
            '67c',
            'f08',
            'b5c',
        ]),
        rt.ExpSubview('dev', [
            '8f4',
            '7f4',
            '059',
            'd08',
        ]),
        rt.ExpSubview('validated', [
            '8f4',
        ]),
    ]

    def load(self):
        reader = common_voice.CommonVoiceReader()
        return reader.load(self.SAMPLE_PATH)
