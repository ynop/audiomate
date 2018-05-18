import pytest

from audiomate.utils import misc


class TestMisc:

    @pytest.mark.parametrize('first_start,first_end,second_start,second_end,overlap', [
        (1.5, 4.2, 3.5, 7.0, 0.7),
        (1.5, 4.2, 0.4, 2.0, 0.5),
        (1.5, 4.2, 1.7, 2.24, 0.54),
        (1.5, 4.2, 0.3, 7.3, 2.7),
        (1.5, 1.7, 1.7, 2.24, 0.0),
        (2.8, 2.9, 1.7, 2.24, 0.0)
    ])
    def test_length_of_overlap(self, first_start, first_end, second_start, second_end, overlap):
        assert misc.length_of_overlap(first_start, first_end, second_start, second_end) == pytest.approx(overlap)
