import unittest

import numpy as np
import librosa

from pingu.corpus import assets

from tests import resources


class UtteranceTest(unittest.TestCase):
    def setUp(self):
        self.file = assets.File('wav', resources.get_wav_file_path('wav_1.wav'))
        self.issuer = assets.Issuer('toni')
        self.utt = assets.Utterance('test', self.file, issuer=self.issuer, start=1.25, end=1.30)

        self.ll_1 = assets.LabelList(idx='alpha', labels=[
            assets.Label('a', 3.2, 4.5),
            assets.Label('b', 5.1, 8.9),
            assets.Label('c', 7.2, 10.5),
            assets.Label('d', 10.5, 14),
            assets.Label('d', 15, 18)
        ])

        self.ll_2 = assets.LabelList(idx='bravo', labels=[
            assets.Label('a', 1.0, 4.2),
            assets.Label('e', 4.2, 7.9),
            assets.Label('c', 7.2, 10.5),
            assets.Label('f', 10.5, 14),
            assets.Label('d', 15, 17.3)
        ])

        self.ll_3 = assets.LabelList(idx='charlie', labels=[
            assets.Label('a', 1.0, 4.2),
            assets.Label('g', 4.2, 7.9)
        ])

        self.utt.set_label_list(self.ll_1)
        self.utt.set_label_list(self.ll_2)
        self.utt.set_label_list(self.ll_3)

    def test_issuer_relation_on_creation(self):
        assert self.utt.issuer == self.issuer
        assert self.utt in self.issuer.utterances

    def test_set_label_list(self):
        assert len(self.utt.label_lists) == 3
        assert self.utt.label_lists['alpha'] == self.ll_1
        assert self.utt.label_lists['bravo'] == self.ll_2
        assert self.utt.label_lists['charlie'] == self.ll_3
        assert self.ll_1.utterance == self.utt
        assert self.ll_2.utterance == self.utt
        assert self.ll_3.utterance == self.utt

    def test_label_values(self):
        assert self.utt.all_label_values() == set(['a', 'b', 'c', 'd', 'e', 'f', 'g'])

    def test_label_values_with_idx_restriction(self):
        all_values = self.utt.all_label_values(label_list_ids=['bravo', 'charlie'])
        assert all_values == set(['a', 'c', 'd', 'e', 'f', 'g'])

    def test_label_count(self):
        assert self.utt.label_count() == {'a': 3, 'b': 1, 'c': 2, 'd': 3, 'e': 1, 'f': 1, 'g': 1}

    def test_label_count_with_idx_restriction(self):
        count = self.utt.label_count(label_list_ids=['bravo', 'charlie'])
        assert count == {'a': 2, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 1}

    def test_read_samples(self):
        expected, __ = librosa.core.load(self.file.path, sr=None, offset=1.25, duration=0.05)
        assert np.array_equal(self.utt.read_samples(), expected)
