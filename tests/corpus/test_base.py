import unittest

import pytest

import audiomate

from tests import resources


class CorpusViewTest(unittest.TestCase):
    def setUp(self):
        self.ds = resources.create_multi_label_corpus()

    def test_all_label_values(self):
        assert self.ds.all_label_values() == {'music', 'speech'}

    def test_label_count(self):
        assert self.ds.label_count() == {'music': 11, 'speech': 7}

    def test_stats(self):
        ds = audiomate.Corpus.load(resources.sample_corpus_path('default'), reader='default')
        stats = ds.stats()

        assert stats.min == pytest.approx(-1.0)
        assert stats.max == pytest.approx(0.99996948)
        assert stats.mean == pytest.approx(-0.00013355668)
        assert stats.var == pytest.approx(0.015060359)

    def test_stats_per_utterance(self):
        ds = audiomate.Corpus.load(resources.sample_corpus_path('default'), reader='default')
        stats = ds.stats_per_utterance()

        assert set(list(stats.keys())) == {'utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'}

        assert stats['utt-1'].min == pytest.approx(-1.0)
        assert stats['utt-1'].max == pytest.approx(0.99996948)
        assert stats['utt-1'].mean == pytest.approx(-0.00023601724)
        assert stats['utt-1'].var == pytest.approx(0.017326673)
        assert stats['utt-1'].num == 118240

        assert stats['utt-3'].min == pytest.approx(-0.92578125)
        assert stats['utt-3'].max == pytest.approx(0.99996948)
        assert stats['utt-3'].mean == pytest.approx(-0.00041901905)
        assert stats['utt-3'].var == pytest.approx(0.017659103)
        assert stats['utt-3'].num == 24000

    def test_label_duration(self):
        durations = self.ds.label_durations()

        assert durations['music'] == pytest.approx(44.0)
        assert durations['speech'] == pytest.approx(45.0)

    def test_duration(self):
        duration = self.ds.total_duration

        assert duration == pytest.approx(85.190375)
