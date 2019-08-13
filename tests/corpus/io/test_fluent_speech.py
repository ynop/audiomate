import os

from audiomate import corpus
from audiomate import issuers
from audiomate.corpus.io import fluent_speech

from tests import resources
from . import reader_test as rt


class TestFreeSpokenDigitReader(rt.CorpusReaderTest):

    SAMPLE_PATH = resources.sample_corpus_path('fluent_speech')
    FILE_TRACK_BASE_PATH = os.path.join(SAMPLE_PATH, 'wavs', 'speakers')

    EXPECTED_NUMBER_OF_TRACKS = 8
    EXPECTED_TRACKS = [
        rt.ExpFileTrack('x', 'a/x.wav'),
        rt.ExpFileTrack('y', 'a/y.wav'),
        rt.ExpFileTrack('z', 'b/z.wav'),
        rt.ExpFileTrack('p', 'g/p.wav'),
        rt.ExpFileTrack('m', 'g/m.wav'),
        rt.ExpFileTrack('i', 'e/i.wav'),
        rt.ExpFileTrack('o', 'f/o.wav'),
        rt.ExpFileTrack('k', 'f/k.wav'),
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 5
    EXPECTED_ISSUERS = [
        rt.ExpSpeaker('a', 2, issuers.Gender.MALE, issuers.AgeGroup.ADULT, 'eng'),
        rt.ExpSpeaker('b', 1, issuers.Gender.MALE, issuers.AgeGroup.SENIOR, 'eng'),
        rt.ExpSpeaker('e', 1, issuers.Gender.MALE, issuers.AgeGroup.ADULT, 'eng'),
        rt.ExpSpeaker('f', 2, issuers.Gender.MALE, issuers.AgeGroup.ADULT, 'fra'),
        rt.ExpSpeaker('g', 2, issuers.Gender.FEMALE, issuers.AgeGroup.ADULT, 'tel'),
    ]

    EXPECTED_NUMBER_OF_UTTERANCES = 8
    EXPECTED_UTTERANCES = [
        rt.ExpUtterance('x', 'x', 'a', 0, float('inf')),
        rt.ExpUtterance('y', 'y', 'a', 0, float('inf')),
        rt.ExpUtterance('z', 'z', 'b', 0, float('inf')),
        rt.ExpUtterance('p', 'p', 'g', 0, float('inf')),
        rt.ExpUtterance('m', 'p', 'g', 0, float('inf')),
        rt.ExpUtterance('i', 'i', 'e', 0, float('inf')),
        rt.ExpUtterance('o', 'o', 'f', 0, float('inf')),
        rt.ExpUtterance('k', 'k', 'f', 0, float('inf')),
    ]

    EXPECTED_LABEL_LISTS = {
        'i': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
            rt.ExpLabelList('action', 1),
            rt.ExpLabelList('location', 1),
        ],
        'o': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
            rt.ExpLabelList('action', 1),
            rt.ExpLabelList('object', 1),
            rt.ExpLabelList('location', 1),
        ],
        'k': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
            rt.ExpLabelList('action', 1),
            rt.ExpLabelList('object', 1),
        ],
        'x': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
            rt.ExpLabelList('action', 1),
        ],
        'y': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
            rt.ExpLabelList('action', 1),
            rt.ExpLabelList('object', 1),
        ],
        'z': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
            rt.ExpLabelList('action', 1),
            rt.ExpLabelList('object', 1),
        ],
        'p': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
            rt.ExpLabelList('action', 1),
            rt.ExpLabelList('object', 1),
            rt.ExpLabelList('location', 1),
        ],
        'm': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
            rt.ExpLabelList('action', 1),
            rt.ExpLabelList('object', 1),
        ],
    }

    EXPECTED_LABELS = {
        'i': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'Change', 0, float('inf')),
            rt.ExpLabel('action', 'change', 0, float('inf')),
            rt.ExpLabel('location', 'kitchen', 0, float('inf')),
        ],
        'o': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'Resume', 0, float('inf')),
            rt.ExpLabel('action', 'activate', 0, float('inf')),
            rt.ExpLabel('object', 'music', 0, float('inf')),
            rt.ExpLabel('location', 'kitchen', 0, float('inf')),
        ],
        'k': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'Turn off the lights', 0, float('inf')),
            rt.ExpLabel('action', 'activate', 0, float('inf')),
            rt.ExpLabel('object', 'lights', 0, float('inf')),
        ],
        'x': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'Change language', 0, float('inf')),
            rt.ExpLabel('action', 'change language', 0, float('inf')),
        ],
        'y': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'Resume', 0, float('inf')),
            rt.ExpLabel('action', 'activate', 0, float('inf')),
            rt.ExpLabel('object', 'music', 0, float('inf')),
        ],
        'z': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'Turn the lights on', 0, float('inf')),
            rt.ExpLabel('action', 'activate', 0, float('inf')),
            rt.ExpLabel('object', 'lights', 0, float('inf')),
        ],
        'p': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'lights', 0, float('inf')),
            rt.ExpLabel('action', 'change', 0, float('inf')),
            rt.ExpLabel('object', 'heat', 0, float('inf')),
            rt.ExpLabel('location', 'bedroom', 0, float('inf')),
        ],
        'm': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'lamp', 0, float('inf')),
            rt.ExpLabel('action', 'activate', 0, float('inf')),
            rt.ExpLabel('object', 'music', 0, float('inf')),
        ],
    }

    EXPECTED_NUMBER_OF_SUBVIEWS = 3
    EXPECTED_SUBVIEWS = [
        rt.ExpSubview('test', [
            'i',
            'o',
            'k',
        ]),
        rt.ExpSubview('train', [
            'x',
            'y',
            'z',
        ]),
        rt.ExpSubview('valid', [
            'p',
            'm',
        ]),
    ]

    def load(self):
        reader = fluent_speech.FluentSpeechReader()
        return reader.load(self.SAMPLE_PATH)
