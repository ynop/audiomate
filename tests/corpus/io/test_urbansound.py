import os

from audiomate import corpus
from audiomate.corpus import io

from tests import resources
from . import reader_test as rt


class TestUrbansound8kReader(rt.CorpusReaderTest):

    SAMPLE_PATH = resources.sample_corpus_path('urbansound8k')
    FILE_TRACK_BASE_PATH = os.path.join(SAMPLE_PATH, 'audio')

    EXPECTED_NUMBER_OF_TRACKS = 5
    EXPECTED_TRACKS = [
        rt.ExpFileTrack('100032-3-0-0', os.path.join('fold5', '100032-3-0-0.wav')),
        rt.ExpFileTrack('100263-2-0-117', os.path.join('fold5', '100263-2-0-117.wav')),
        rt.ExpFileTrack('145612-6-3-0', os.path.join('fold8', '145612-6-3-0.wav')),
        rt.ExpFileTrack('145683-6-5-0', os.path.join('fold9', '145683-6-5-0.wav')),
        rt.ExpFileTrack('79377-9-0-4', os.path.join('fold2', '79377-9-0-4.wav')),
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 0

    EXPECTED_NUMBER_OF_UTTERANCES = 5
    EXPECTED_UTTERANCES = [
        rt.ExpUtterance('100032-3-0-0', '100032-3-0-0', None, 0, float('inf')),
        rt.ExpUtterance('100263-2-0-117', '100263-2-0-117', None, 0, float('inf')),
        rt.ExpUtterance('145612-6-3-0', '145612-6-3-0', None, 0, float('inf')),
        rt.ExpUtterance('145683-6-5-0', '145683-6-5-0', None, 0, float('inf')),
        rt.ExpUtterance('79377-9-0-4', '79377-9-0-4', None, 0, float('inf')),
    ]

    EXPECTED_LABEL_LISTS = {
        '100032-3-0-0': [rt.ExpLabelList(corpus.LL_SOUND_CLASS, 1)],
        '100263-2-0-117': [rt.ExpLabelList(corpus.LL_SOUND_CLASS, 1)],
        '145612-6-3-0': [rt.ExpLabelList(corpus.LL_SOUND_CLASS, 1)],
        '145683-6-5-0': [rt.ExpLabelList(corpus.LL_SOUND_CLASS, 1)],
        '79377-9-0-4': [rt.ExpLabelList(corpus.LL_SOUND_CLASS, 1)],
    }

    EXPECTED_LABELS = {
        '100032-3-0-0':   [rt.ExpLabel(corpus.LL_SOUND_CLASS, 'dog_bark', 0, float('inf'))],
        '100263-2-0-117': [rt.ExpLabel(corpus.LL_SOUND_CLASS, 'children_playing', 0, float('inf'))],
        '145612-6-3-0':   [rt.ExpLabel(corpus.LL_SOUND_CLASS, 'gun_shot', 0, float('inf'))],
        '145683-6-5-0':   [rt.ExpLabel(corpus.LL_SOUND_CLASS, 'gun_shot', 0, float('inf'))],
        '79377-9-0-4':    [rt.ExpLabel(corpus.LL_SOUND_CLASS, 'street_music', 0, float('inf'))],
    }

    EXPECTED_NUMBER_OF_SUBVIEWS = 4
    EXPECTED_SUBVIEWS = [
        rt.ExpSubview('fold2', [
            '79377-9-0-4',
        ]),
        rt.ExpSubview('fold5', [
            '100032-3-0-0',
            '100263-2-0-117',
        ]),
        rt.ExpSubview('fold8', [
            '145612-6-3-0',
        ]),
        rt.ExpSubview('fold9', [
            '145683-6-5-0',
        ]),
    ]

    def load(self):
        return io.Urbansound8kReader().load(self.SAMPLE_PATH)
