import unittest

import pytest

import pingu

from tests import resources


class CorpusViewTest(unittest.TestCase):
    def setUp(self):
        self.ds = resources.create_multi_label_corpus()

    def test_all_label_values(self):
        assert self.ds.all_label_values() == set(['music', 'speech'])

    def test_label_count(self):
        assert self.ds.label_count() == {'music': 11, 'speech': 7}

    def test_stats(self):
        ds = pingu.Corpus.load(resources.sample_default_ds_path(), reader='default')
        stats = ds.stats()

        assert stats[0] == pytest.approx(-1.0)
        assert stats[1] == pytest.approx(0.99996948)
        assert stats[2] == pytest.approx(-0.00013355668)
        assert stats[3] == pytest.approx(0.015060359)

    def test_stats_per_utterance(self):
        ds = pingu.Corpus.load(resources.sample_default_ds_path(), reader='default')
        stats = ds.stats_per_utterance()

        assert set(list(stats.keys())) == set(['utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'])

        assert stats['utt-1'][0] == pytest.approx(-1.0)
        assert stats['utt-1'][1] == pytest.approx(0.99996948)
        assert stats['utt-1'][2] == pytest.approx(-0.00023601724)
        assert stats['utt-1'][3] == pytest.approx(0.017326673)
        assert stats['utt-1'][4] == 118240

        assert stats['utt-3'][0] == pytest.approx(-0.92578125)
        assert stats['utt-3'][1] == pytest.approx(0.99996948)
        assert stats['utt-3'][2] == pytest.approx(-0.00041901905)
        assert stats['utt-3'][3] == pytest.approx(0.017659103)
        assert stats['utt-3'][4] == 24000
