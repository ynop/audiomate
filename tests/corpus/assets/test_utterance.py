import unittest

import numpy as np
import librosa
import pytest

from audiomate.corpus import assets

from tests import resources


class UtteranceTest(unittest.TestCase):
    def setUp(self):
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

        self.ll_duplicate_idx = assets.LabelList(idx='charlie', labels=[
            assets.Label('t', 1.0, 4.2),
            assets.Label('h', 4.2, 7.9)
        ])

        self.ll_3 = assets.LabelList(idx='charlie', labels=[
            assets.Label('a', 1.0, 4.2),
            assets.Label('g', 4.2, 7.9)
        ])

        self.file = assets.File('wav', resources.sample_wav_file('wav_1.wav'))
        self.issuer = assets.Issuer('toni')
        self.utt = assets.Utterance('test', self.file, issuer=self.issuer, start=1.25, end=1.30, label_lists=[
            self.ll_1,
            self.ll_2,
            self.ll_duplicate_idx,
            self.ll_3
        ])

    def test_end_abs(self):
        assert self.utt.end_abs == 1.30

    def test_end_abs_end_of_file(self):
        utt = assets.Utterance('utt', self.file, start=0.3, end=-1)
        assert utt.end_abs == pytest.approx(2.5951875)

    def test_duration(self):
        assert self.utt.duration == pytest.approx(0.05)

    def test_duration_end_of_file(self):
        utt = assets.Utterance('utt', self.file, start=0.3, end=-1)
        assert utt.duration == pytest.approx(2.2951875)

    def test_issuer_relation_on_creation(self):
        assert self.utt.issuer == self.issuer
        assert self.utt in self.issuer.utterances

    def test_set_label_list(self):
        ll_4 = assets.LabelList(idx='delta', labels=[
            assets.Label('y', 0.0, 3.3),
            assets.Label('t', 3.8, 7.9)
        ])

        self.utt.set_label_list(ll_4)

        assert len(self.utt.label_lists) == 4
        assert self.utt.label_lists['alpha'] == self.ll_1
        assert self.utt.label_lists['bravo'] == self.ll_2
        assert self.utt.label_lists['charlie'] == self.ll_3
        assert self.utt.label_lists['delta'] == ll_4
        assert self.ll_1.utterance == self.utt
        assert self.ll_2.utterance == self.utt
        assert self.ll_3.utterance == self.utt
        assert ll_4.utterance == self.utt

    def test_label_values(self):
        assert self.utt.all_label_values() == {'a', 'b', 'c', 'd', 'e', 'f', 'g'}

    def test_label_values_with_idx_restriction(self):
        all_values = self.utt.all_label_values(label_list_ids=['bravo', 'charlie'])
        assert all_values == {'a', 'c', 'd', 'e', 'f', 'g'}

    def test_label_count(self):
        assert self.utt.label_count() == {'a': 3, 'b': 1, 'c': 2, 'd': 3, 'e': 1, 'f': 1, 'g': 1}

    def test_label_count_with_idx_restriction(self):
        count = self.utt.label_count(label_list_ids=['bravo', 'charlie'])
        assert count == {'a': 2, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 1}

    def test_label_duration(self):
        duration = self.utt.label_total_duration(label_list_ids=['alpha', 'charlie'])
        assert duration['a'] == pytest.approx(4.5)
        assert duration['b'] == pytest.approx(3.8)
        assert duration['c'] == pytest.approx(3.3)
        assert duration['d'] == pytest.approx(6.5)
        assert duration['g'] == pytest.approx(3.7)

    def test_read_samples(self):
        expected, __ = librosa.core.load(self.file.path, sr=None, offset=1.25, duration=0.05)
        assert np.array_equal(self.utt.read_samples(), expected)

    def test_read_samples_with_offset(self):
        expected, __ = librosa.core.load(self.file.path, sr=None, offset=1.27, duration=0.03)
        assert np.array_equal(self.utt.read_samples(offset=0.02), expected)

    def test_read_samples_with_duration(self):
        expected, __ = librosa.core.load(self.file.path, sr=None, offset=1.26, duration=0.03)
        assert np.array_equal(self.utt.read_samples(offset=0.01, duration=0.03), expected)

    def test_num_samples(self):
        assert self.utt.num_samples() == 800

    def test_num_samples_non_native_sr(self):
        assert self.utt.num_samples(sr=12000) == 600

    def test_num_samples_matches_read_samples(self):
        assert self.utt.read_samples().shape[0] == self.utt.num_samples()
        assert self.utt.read_samples(sr=11255).shape[0] == self.utt.num_samples(sr=11255)
