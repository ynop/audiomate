import unittest

import pytest

from pingu.corpus import assets

from tests import resources


class FeatureContainerTest(unittest.TestCase):
    def setUp(self):
        self.container = assets.FeatureContainer(resources.get_feat_container_path())
        self.container.open()

    def tearDown(self):
        self.container.close()

    def test_stats_per_utterance(self):
        utt_stats = self.container.stats_per_utterance()

        assert utt_stats['utt-1'][0] == pytest.approx(0.0071605651933048797)
        assert utt_stats['utt-1'][1] == pytest.approx(0.9967182746569494)
        assert utt_stats['utt-1'][2] == pytest.approx(0.51029100520776705)
        assert utt_stats['utt-1'][3] == pytest.approx(0.079222738766221268)
        assert utt_stats['utt-1'][4] == 100

        assert utt_stats['utt-2'][0] == pytest.approx(0.01672865642756316)
        assert utt_stats['utt-2'][1] == pytest.approx(0.99394433783429104)
        assert utt_stats['utt-2'][2] == pytest.approx(0.46471979908661543)
        assert utt_stats['utt-2'][3] == pytest.approx(0.066697466410977804)
        assert utt_stats['utt-2'][4] == 65

        assert utt_stats['utt-3'][0] == pytest.approx(0.014999482706963607)
        assert utt_stats['utt-3'][1] == pytest.approx(0.99834417857609881)
        assert utt_stats['utt-3'][2] == pytest.approx(0.51042690965262705)
        assert utt_stats['utt-3'][3] == pytest.approx(0.071833200069641057)
        assert utt_stats['utt-3'][4] == 220

    def test_stats(self):
        stats = self.container.stats()

        assert stats[0] == pytest.approx(0.0071605651933048797)
        assert stats[1] == pytest.approx(0.99834417857609881)
        assert stats[2] == pytest.approx(0.50267482489606408)
        assert stats[3] == pytest.approx(0.07317811077366114)
