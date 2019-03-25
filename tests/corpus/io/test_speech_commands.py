import os

from audiomate import corpus
from audiomate import issuers
from audiomate.corpus import io

from tests import resources
from . import reader_test as rt


class TestSpeechCommandsReader(rt.CorpusReaderTest):

    SAMPLE_PATH = resources.sample_corpus_path('speech_commands')
    FILE_TRACK_BASE_PATH = SAMPLE_PATH

    EXPECTED_NUMBER_OF_TRACKS = 13
    EXPECTED_TRACKS = [
        rt.ExpFileTrack('0b77ee66_nohash_0_bed', os.path.join('bed', '0b77ee66_nohash_0.wav')),
        rt.ExpFileTrack('0b77ee66_nohash_1_bed', os.path.join('bed', '0b77ee66_nohash_1.wav')),
        rt.ExpFileTrack('0b77ee66_nohash_2_bed', os.path.join('bed', '0b77ee66_nohash_2.wav')),
        rt.ExpFileTrack('0bde966a_nohash_0_bed', os.path.join('bed', '0bde966a_nohash_0.wav')),
        rt.ExpFileTrack('0bde966a_nohash_1_bed', os.path.join('bed', '0bde966a_nohash_1.wav')),
        rt.ExpFileTrack('0c40e715_nohash_0_bed', os.path.join('bed', '0c40e715_nohash_0.wav')),
        rt.ExpFileTrack('d5c41d6a_nohash_0_marvin', os.path.join('marvin', 'd5c41d6a_nohash_0.wav')),
        rt.ExpFileTrack('d7a58714_nohash_0_marvin', os.path.join('marvin', 'd7a58714_nohash_0.wav')),
        rt.ExpFileTrack('d8a5ace5_nohash_0_marvin', os.path.join('marvin', 'd8a5ace5_nohash_0.wav')),
        rt.ExpFileTrack('0a7c2a8d_nohash_0_one', os.path.join('one', '0a7c2a8d_nohash_0.wav')),
        rt.ExpFileTrack('0b77ee66_nohash_0_one', os.path.join('one', '0b77ee66_nohash_0.wav')),
        rt.ExpFileTrack('c1b7c224_nohash_0_one', os.path.join('one', 'c1b7c224_nohash_0.wav')),
        rt.ExpFileTrack('c1b7c224_nohash_1_one', os.path.join('one', 'c1b7c224_nohash_1.wav')),
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 8
    EXPECTED_ISSUERS = [
        rt.ExpSpeaker('0b77ee66', 4, issuers.Gender.UNKNOWN, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('0bde966a', 2, issuers.Gender.UNKNOWN, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('0c40e715', 1, issuers.Gender.UNKNOWN, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('d5c41d6a', 1, issuers.Gender.UNKNOWN, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('d7a58714', 1, issuers.Gender.UNKNOWN, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('d8a5ace5', 1, issuers.Gender.UNKNOWN, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('0a7c2a8d', 1, issuers.Gender.UNKNOWN, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('c1b7c224', 2, issuers.Gender.UNKNOWN, issuers.AgeGroup.UNKNOWN, None),
    ]

    EXPECTED_NUMBER_OF_UTTERANCES = 13
    EXPECTED_UTTERANCES = [
        rt.ExpUtterance('0b77ee66_nohash_0_bed', '0b77ee66_nohash_0_bed', '0b77ee66', 0, float('inf')),
        rt.ExpUtterance('0b77ee66_nohash_1_bed', '0b77ee66_nohash_1_bed', '0b77ee66', 0, float('inf')),
        rt.ExpUtterance('0b77ee66_nohash_2_bed', '0b77ee66_nohash_2_bed', '0b77ee66', 0, float('inf')),
        rt.ExpUtterance('0bde966a_nohash_0_bed', '0bde966a_nohash_0_bed', '0bde966a', 0, float('inf')),
        rt.ExpUtterance('0bde966a_nohash_1_bed', '0bde966a_nohash_1_bed', '0bde966a', 0, float('inf')),
        rt.ExpUtterance('0c40e715_nohash_0_bed', '0c40e715_nohash_0_bed', '0c40e715', 0, float('inf')),
        rt.ExpUtterance('d5c41d6a_nohash_0_marvin', 'd5c41d6a_nohash_0_marvin', 'd5c41d6a', 0, float('inf')),
        rt.ExpUtterance('d7a58714_nohash_0_marvin', 'd7a58714_nohash_0_marvin', 'd7a58714', 0, float('inf')),
        rt.ExpUtterance('d8a5ace5_nohash_0_marvin', 'd8a5ace5_nohash_0_marvin', 'd8a5ace5', 0, float('inf')),
        rt.ExpUtterance('0a7c2a8d_nohash_0_one', '0a7c2a8d_nohash_0_one', '0a7c2a8d', 0, float('inf')),
        rt.ExpUtterance('0b77ee66_nohash_0_one', '0b77ee66_nohash_0_one', '0b77ee66', 0, float('inf')),
        rt.ExpUtterance('c1b7c224_nohash_0_one', 'c1b7c224_nohash_0_one', 'c1b7c224', 0, float('inf')),
        rt.ExpUtterance('c1b7c224_nohash_1_one', 'c1b7c224_nohash_1_one', 'c1b7c224', 0, float('inf')),
    ]

    EXPECTED_LABEL_LISTS = {
        '0b77ee66_nohash_0_bed': [rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1)],
        '0b77ee66_nohash_1_bed': [rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1)],
        '0b77ee66_nohash_2_bed': [rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1)],
        '0bde966a_nohash_0_bed': [rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1)],
        '0bde966a_nohash_1_bed': [rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1)],
        '0c40e715_nohash_0_bed': [rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1)],
        'd5c41d6a_nohash_0_marvin': [rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1)],
        'd7a58714_nohash_0_marvin': [rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1)],
        'd8a5ace5_nohash_0_marvin': [rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1)],
        '0a7c2a8d_nohash_0_one': [rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1)],
        '0b77ee66_nohash_0_one': [rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1)],
        'c1b7c224_nohash_0_one': [rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1)],
        'c1b7c224_nohash_1_one': [rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1)],
    }

    EXPECTED_LABELS = {
        '0b77ee66_nohash_0_bed': [rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'bed', 0, float('inf'))],
        '0b77ee66_nohash_1_bed': [rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'bed', 0, float('inf'))],
        '0b77ee66_nohash_2_bed': [rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'bed', 0, float('inf'))],
        '0bde966a_nohash_0_bed': [rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'bed', 0, float('inf'))],
        '0bde966a_nohash_1_bed': [rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'bed', 0, float('inf'))],
        '0c40e715_nohash_0_bed': [rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'bed', 0, float('inf'))],
        'd5c41d6a_nohash_0_marvin': [rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'marvin', 0, float('inf'))],
        'd7a58714_nohash_0_marvin': [rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'marvin', 0, float('inf'))],
        'd8a5ace5_nohash_0_marvin': [rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'marvin', 0, float('inf'))],
        '0a7c2a8d_nohash_0_one': [rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'one', 0, float('inf'))],
        '0b77ee66_nohash_0_one': [rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'one', 0, float('inf'))],
        'c1b7c224_nohash_0_one': [rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'one', 0, float('inf'))],
        'c1b7c224_nohash_1_one': [rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'one', 0, float('inf'))],
    }

    EXPECTED_NUMBER_OF_SUBVIEWS = 3
    EXPECTED_SUBVIEWS = [
        rt.ExpSubview('train', [
            '0b77ee66_nohash_0_bed',
            '0b77ee66_nohash_1_bed',
            '0b77ee66_nohash_2_bed',
            'd5c41d6a_nohash_0_marvin',
            'c1b7c224_nohash_0_one',
            'c1b7c224_nohash_1_one',
        ]),
        rt.ExpSubview('dev', [
            '0c40e715_nohash_0_bed',
            'd8a5ace5_nohash_0_marvin',
            '0a7c2a8d_nohash_0_one',
        ]),
        rt.ExpSubview('test', [
            '0bde966a_nohash_0_bed',
            '0bde966a_nohash_1_bed',
            'd7a58714_nohash_0_marvin',
            '0b77ee66_nohash_0_one',
        ]),
    ]

    def load(self):
        return io.SpeechCommandsReader().load(self.SAMPLE_PATH)
