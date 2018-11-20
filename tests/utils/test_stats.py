import pytest

import numpy as np

from audiomate.utils import stats


class TestDataStats:

    def test_values(self):
        s = stats.DataStats(2.3, 1.2, 4.0, -2, 99)
        assert np.array_equal(s.values, np.array([2.3, 1.2, 4.0, -2, 99]))

    def test_concatenate(self):
        values = np.random.randn(20).reshape(4, 5)

        stats_list = []

        for x in values:
            s = stats.DataStats(
                float(x.mean()),
                float(x.var()),
                x.min(),
                x.max(),
                x.size
            )
            stats_list.append(s)

        concatenated = stats.DataStats.concatenate(stats_list)

        assert concatenated.mean == pytest.approx(np.mean(values))
        assert concatenated.var == pytest.approx(np.var(values))
        assert concatenated.min == pytest.approx(np.min(values))
        assert concatenated.max == pytest.approx(np.max(values))
        assert concatenated.num == values.size

    def test_to_dict(self):
        s = stats.DataStats(2.3, 1.2, -2, 4.0, 99)
        d = s.to_dict()

        assert d == {'mean': 2.3, 'var': 1.2, 'max': 4.0, 'min': -2, 'num': 99}

    def test_from_dict(self):
        s = stats.DataStats.from_dict({
            'mean': 2.3,
            'var': 1.2,
            'max': 4.0,
            'min': -2,
            'num': 99
        })

        assert s.mean == pytest.approx(2.3)
        assert s.var == pytest.approx(1.2)
        assert s.min == pytest.approx(-2)
        assert s.max == pytest.approx(4.0)
        assert s.num == 99
