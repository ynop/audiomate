import unittest
import pytest

import numpy as np

from pingu.utils import stats


class DataStatsTest(unittest.TestCase):

    def test_values(self):
        s = stats.DataStats(2.3, 1.2, 4.0, -2, 99)
        assert np.array_equal(s.values, np.array([2.3, 1.2, 4.0, -2, 99]))

    def test_concatenate(self):
        values = np.random.randn(20).reshape(4, 5)
        stats_list = [stats.DataStats(float(x.mean()), float(x.var()), x.min(), x.max(), x.size) for x in values]

        concatenated = stats.DataStats.concatenate(stats_list)

        assert concatenated.mean == pytest.approx(np.mean(values))
        assert concatenated.var == pytest.approx(np.var(values))
        assert concatenated.min == pytest.approx(np.min(values))
        assert concatenated.max == pytest.approx(np.max(values))
        assert concatenated.num == values.size
