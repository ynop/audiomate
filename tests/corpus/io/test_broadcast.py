import os

from audiomate import issuers
from audiomate.corpus import io

from tests import resources
from . import reader_test as rt


class TestBroadcastReader(rt.CorpusReaderTest):

    SAMPLE_PATH = resources.sample_corpus_path('broadcast')
    FILE_TRACK_BASE_PATH = os.path.join(SAMPLE_PATH, 'files')

    EXPECTED_NUMBER_OF_TRACKS = 4
    EXPECTED_TRACKS = [
        rt.ExpFileTrack('file-1', os.path.join('a', 'wav_1.wav')),
        rt.ExpFileTrack('file-2', os.path.join('b', 'wav_2.wav')),
        rt.ExpFileTrack('file-3', os.path.join('c', 'wav_3.wav')),
        rt.ExpFileTrack('file-4', os.path.join('d', 'wav_4.wav')),
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 3
    EXPECTED_ISSUERS = [
        rt.ExpSpeaker('speaker-1', 2, issuers.Gender.FEMALE, issuers.AgeGroup.CHILD, 'eng'),
        rt.ExpArtist('speaker-2', 2, None),
        rt.ExpIssuer('speaker-3', 1),
    ]

    EXPECTED_NUMBER_OF_UTTERANCES = 5
    EXPECTED_UTTERANCES = [
        rt.ExpUtterance('utt-1', 'file-1', 'speaker-1', 0, float('inf')),
        rt.ExpUtterance('utt-2', 'file-2', 'speaker-1', 0, float('inf')),
        rt.ExpUtterance('utt-3', 'file-3', 'speaker-2', 0, 100),
        rt.ExpUtterance('utt-4', 'file-3', 'speaker-2', 100, 150),
        rt.ExpUtterance('utt-5', 'file-4', 'speaker-3', 0, float('inf')),
    ]

    EXPECTED_LABEL_LISTS = {
        'utt-1': [
            rt.ExpLabelList('jingles', 2),
            rt.ExpLabelList('music', 2),
        ],
        'utt-2': [
            rt.ExpLabelList('default', 2),
        ],
        'utt-3': [
            rt.ExpLabelList('default', 2),
        ],
        'utt-4': [
            rt.ExpLabelList('default', 2),
        ],
        'utt-5': [
            rt.ExpLabelList('default', 2),
        ],
    }

    EXPECTED_LABELS = {
        'utt-1': [
            rt.ExpLabel('jingles', 'velo', 80, 82.4),
        ],
        'utt-4': [
            rt.ExpLabel('default', 'hallo', 105, 130.5),
        ],
    }

    EXPECTED_NUMBER_OF_SUBVIEWS = 0

    def test_load_label_meta(self):
        ds = self.load()

        utt_1 = ds.utterances['utt-1']
        labels = sorted(utt_1.label_lists['jingles'])

        assert len(labels[0].meta) == 0
        assert len(labels[1].meta) == 3
        assert labels[1].meta['lang'] == 'de'
        assert labels[1].meta['prio'] == 4
        assert labels[1].meta['unique']

    def load(self):
        return io.BroadcastReader().load(self.SAMPLE_PATH)
