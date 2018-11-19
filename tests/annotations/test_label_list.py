import pytest

from audiomate import tracks
from audiomate.annotations import Label, LabelList

from tests import resources


@pytest.fixture
def sample_label():
    file_track = tracks.FileTrack('wav', resources.sample_wav_file('wav_1.wav'))
    utt = tracks.Utterance('utt', file_track, start=0.3, end=-1)
    test_label = Label('a', start=0.5, end=-1)
    ll = LabelList()
    ll.append(test_label)
    utt.set_label_list(ll)

    return test_label


class TestLabelList:

    def test_start_abs(self, sample_label):
        assert sample_label.start_abs == pytest.approx(0.8)

    def test_end_abs(self, sample_label):
        assert sample_label.end_abs == pytest.approx(2.5951875)

    def test_duration(self, sample_label):
        assert sample_label.duration == pytest.approx(1.7951875)

    def test_append(self):
        ll = LabelList()

        label = Label('some text')
        ll.append(label)

        assert len(ll) == 1
        assert label.label_list == ll

    def test_extend(self):
        ll = LabelList()

        label_a = Label('some text')
        label_b = Label('more text')
        label_c = Label('text again')
        ll.extend([label_a, label_b, label_c])

        assert len(ll) == 3
        assert label_a.label_list == ll
        assert label_b.label_list == ll
        assert label_c.label_list == ll

    def test_ranges(self):
        ll = LabelList(labels=[
            Label('a', 3.2, 4.5),
            Label('b', 5.1, 8.9),
            Label('c', 7.2, 10.5),
            Label('d', 10.5, 14)
        ])

        ranges = ll.ranges()

        r = next(ranges)
        assert 3.2 == r[0]
        assert 4.5 == r[1]
        assert ll[0] in r[2]

        r = next(ranges)
        assert 5.1 == r[0]
        assert 7.2 == r[1]
        assert ll[1] in r[2]

        r = next(ranges)
        assert 7.2 == r[0]
        assert 8.9 == r[1]
        assert ll[1] in r[2]
        assert ll[2] in r[2]

        r = next(ranges)
        assert 8.9 == r[0]
        assert 10.5 == r[1]
        assert ll[2] in r[2]

        r = next(ranges)
        assert 10.5 == r[0]
        assert 14 == r[1]
        assert ll[3] in r[2]

        with pytest.raises(StopIteration):
            next(ranges)

    def test_ranges_with_empty(self):
        ll = LabelList(labels=[
            Label('a', 3.2, 4.5),
            Label('b', 5.1, 8.9),
            Label('c', 7.2, 10.5),
            Label('d', 10.5, 14)
        ])

        ranges = ll.ranges(yield_ranges_without_labels=True)

        r = next(ranges)
        assert 3.2 == r[0]
        assert 4.5 == r[1]
        assert ll[0] in r[2]

        r = next(ranges)
        assert 4.5 == r[0]
        assert 5.1 == r[1]
        assert 0 == len(r[2])

        r = next(ranges)
        assert 5.1 == r[0]
        assert 7.2 == r[1]
        assert ll[1] in r[2]

        r = next(ranges)
        assert 7.2 == r[0]
        assert 8.9 == r[1]
        assert ll[1] in r[2]
        assert ll[2] in r[2]

        r = next(ranges)
        assert 8.9 == r[0]
        assert 10.5 == r[1]
        assert ll[2] in r[2]

        r = next(ranges)
        assert 10.5 == r[0]
        assert 14 == r[1]
        assert ll[3] in r[2]

        with pytest.raises(StopIteration):
            next(ranges)

    def test_ranges_include_labels(self):
        ll = LabelList(labels=[
            Label('a', 3.2, 4.5),
            Label('b', 5.1, 8.9)
        ])

        ranges = ll.ranges(include_labels=['a'])

        r = next(ranges)
        assert 3.2 == r[0]
        assert 4.5 == r[1]
        assert ll[0] in r[2]

        with pytest.raises(StopIteration):
            next(ranges)

    def test_ranges_zero_to_end(self):
        ll = LabelList(labels=[
            Label('a', 0, -1),
            Label('b', 5.1, 8.9)
        ])

        ranges = ll.ranges()

        r = next(ranges)
        assert 0 == r[0]
        assert 5.1 == r[1]
        assert ll[0] in r[2]

        r = next(ranges)
        assert 5.1 == r[0]
        assert 8.9 == r[1]
        assert ll[0] in r[2]
        assert ll[1] in r[2]

        r = next(ranges)
        assert 8.9 == r[0]
        assert -1 == r[1]
        assert ll[0] in r[2]

        with pytest.raises(StopIteration):
            next(ranges)

    def test_ranges_with_same_start_times(self):
        ll = LabelList(labels=[
            Label('a', 1.2, 1.3),
            Label('b', 1.2, 5.6)
        ])

        ranges = ll.ranges()

        r = next(ranges)
        assert r[0] == 1.2
        assert r[1] == 1.3
        assert len(r[2]) == 2
        assert ll[0] in r[2]
        assert ll[1] in r[2]

        r = next(ranges)
        assert r[0] == 1.3
        assert r[1] == 5.6
        assert len(r[2]) == 1
        assert ll[1] in r[2]

        with pytest.raises(StopIteration):
            next(ranges)

    def test_label_count(self):
        ll = LabelList(labels=[
            Label('a', 3.2, 4.5),
            Label('b', 5.1, 8.9),
            Label('c', 7.2, 10.5),
            Label('a', 10.5, 14),
            Label('c', 13, 14)
        ])

        res = ll.label_count()

        assert 2 == res['a']
        assert 1 == res['b']
        assert 2 == res['c']

    def test_label_total_durations(self):
        ll = LabelList(labels=[
            Label('a', 3.2, 4.5),
            Label('b', 5.1, 8.9),
            Label('c', 7.2, 10.5),
            Label('a', 10.5, 14),
            Label('c', 13, 14)
        ])

        res = ll.label_total_duration()

        assert res['a'] == pytest.approx(4.8)
        assert res['b'] == pytest.approx(3.8)
        assert res['c'] == pytest.approx(4.3)

    def test_create_single(self):
        ll = LabelList.create_single('bob')

        assert len(ll) == 1
        assert ll.idx == 'default'
        assert ll[0].value == 'bob'
        assert ll[0].start == 0
        assert ll[0].end == -1

    def test_create_single_with_custom_idx(self):
        ll = LabelList.create_single('bob', idx='name')

        assert len(ll) == 1
        assert ll.idx == 'name'
        assert ll[0].value == 'bob'
        assert ll[0].start == 0
        assert ll[0].end == -1

    def test_join(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('b', 4.0, 8.0),
            Label('c', 9.0, 12.0)
        ])

        assert ll.join() == 'a b c'

    def test_join_with_custom_delimiter(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('b', 4.0, 8.0),
            Label('c', 9.0, 12.0)
        ])

        assert ll.join(delimiter=' - ') == 'a - b - c'

    def test_join_raises_error_if_overlap_is_higher_than_threshold(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('b', 3.8, 8.0),
            Label('c', 9.0, 12.0)
        ])

        with pytest.raises(ValueError):
            ll.join(overlap_threshold=0.1)

    def test_join_raises_error_if_overlap_is_higher_than_threshold_given_an_endless_label(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('b', 4.5, -1),
            Label('c', 9.0, 12.0)
        ])

        with pytest.raises(ValueError):
            ll.join(overlap_threshold=0.1)

    def test_tokenized(self):
        ll = LabelList(idx='test', labels=[
            Label('a u t', 0.0, 4.0),
            Label('b x', 4.0, 8.0),
            Label('c', 9.0, 12.0)
        ])

        assert ll.tokenized() == ['a', 'u', 't', 'b', 'x', 'c']

    def test_tokenized_raises_error_if_overlap_is_higher_than_threshold(self):
        ll = LabelList(idx='test', labels=[
            Label('a u t', 0.0, 4.0),
            Label('b x', 3.85, 8.0),
            Label('c', 9.0, 12.0)
        ])

        with pytest.raises(ValueError):
            ll.tokenized()

    def test_tokenized_raises_error_if_overlap_is_higher_than_threshold_given_an_endless_label(self):
        ll = LabelList(idx='test', labels=[
            Label('a u t', 0.0, 4.0),
            Label('b x', 4.5, -1),
            Label('c', 9.0, 12.0)
        ])

        with pytest.raises(ValueError):
            ll.tokenized()

    def test_split(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('b', 4.0, 8.0),
            Label('c', 9.0, 12.0)
        ])

        res = ll.split([1.9, 6.2, 10.5])

        assert len(res) == 4

        assert len(res[0]) == 1
        assert res[0][0] == Label('a', 0.0, 1.9)

        assert len(res[1]) == 2
        assert res[1][0] == Label('a', 1.9, 4.0)
        assert res[1][1] == Label('b', 4.0, 6.2)

        assert len(res[2]) == 2
        assert res[2][0] == Label('b', 6.2, 8.0)
        assert res[2][1] == Label('c', 9.0, 10.5)

        assert len(res[3]) == 1
        assert res[3][0] == Label('c', 10.5, 12.0)

    def test_split_unsorted_label_list(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('c', 9.0, 12.0),
            Label('b', 4.0, 8.0)
        ])

        res = ll.split([1.9, 6.2, 10.5])

        assert len(res) == 4

        assert len(res[0]) == 1
        assert res[0][0] == Label('a', 0.0, 1.9)

        assert len(res[1]) == 2
        assert res[1][0] == Label('a', 1.9, 4.0)
        assert res[1][1] == Label('b', 4.0, 6.2)

        assert len(res[2]) == 2
        assert res[2][0] == Label('b', 6.2, 8.0)
        assert res[2][1] == Label('c', 9.0, 10.5)

        assert len(res[3]) == 1
        assert res[3][0] == Label('c', 10.5, 12.0)

    def test_split_label_within_cutting_points_is_included(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('b', 4.0, 8.0),
            Label('c', 9.0, 12.0)
        ])

        res = ll.split([1.9, 10.5])

        assert len(res[1]) == 3
        assert res[1][1].value == 'b'
        assert res[1][1].start == 4.0
        assert res[1][1].end == 8.0

    def test_split_with_endless_label(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('c', 4.0, -1)
        ])

        res = ll.split([1.9, 10.5])

        assert len(res) == 3

        assert len(res[2]) == 1
        assert res[2][0].value == 'c'
        assert res[2][0].start == 10.5
        assert res[2][0].end == -1

    def test_split_with_cutting_point_after_last_label(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('c', 4.0, 8.9)
        ])

        res = ll.split([10.5])

        assert len(res) == 2
        assert len(res[0]) == 2
        assert len(res[1]) == 0

    def test_split_cutting_point_on_boundary_doesnot_split_label(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 9.0),
            Label('c', 9.0, 12.0)
        ])

        res = ll.split([9.0])

        assert len(res) == 2

        assert len(res[0]) == 1
        assert len(res[1]) == 1

    def test_split_without_cutting_points_raises_error(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 9.0),
            Label('c', 9.0, 12.0)
        ])

        with pytest.raises(ValueError):
            ll.split([])

    def test_split_with_shifting_start_and_endtime(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 9.0),
            Label('c', 9.0, 12.0)
        ])

        res = ll.split([4.2], shift_times=True)

        assert len(res) == 2

        assert len(res[0]) == 1
        assert res[0][0].value == 'a'
        assert res[0][0].start == 0.0
        assert res[0][0].end == 4.2

        assert len(res[1]) == 2
        assert res[1][0] == Label('a', 0.0, 4.8)
        assert res[1][1] == Label('c', 4.8, 7.8)

    def test_split_first_label_not_splitted(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 9.0),
            Label('c', 9.0, 12.0)
        ])

        res = ll.split([11.2], shift_times=True)

        assert len(res) == 2

        assert len(res[0]) == 2
        assert res[0][0] == Label('a', 0.0, 9.0)
        assert res[0][1] == Label('c', 9.0, 11.2)

        assert len(res[1]) == 1
        assert res[1][0] == Label('c', 0.0, pytest.approx(0.8))

    def test_split_single_label_that_doesnt_start_at_zero(self):
        ll = LabelList(idx='test', labels=[
            Label('c', 8.0, 12.0)
        ])

        res = ll.split([11.2], shift_times=True)

        assert len(res) == 2

        assert len(res[0]) == 1
        assert res[0][0] == Label('c', 8.0, 11.2)

        assert len(res[1]) == 1
        assert res[1][0] == Label('c', 0.0, pytest.approx(0.8))
