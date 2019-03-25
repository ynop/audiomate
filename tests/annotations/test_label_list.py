import pytest

from audiomate.annotations import Label, LabelList


class TestLabelList:

    def test_equal(self):
        ll_a = LabelList(idx='test', labels=[
            Label('c', 12.05, 14.0),
            Label('a', 0.0, 4.0),
            Label('b', 4.0, 8.0),
            Label('b', 7.9, 9.3),
            Label('c', 9.0, 12.0),
        ])

        ll_b = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('b', 7.9, 9.3),
            Label('c', 9.0, 12.0),
            Label('b', 4.0, 8.0),
            Label('c', 12.05, 14.0),
        ])

        assert ll_a == ll_b

    def test_is(self):
        ll_a = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('b', 4.0, 8.0),
            Label('b', 7.9, 9.3),
            Label('c', 9.0, 12.0),
            Label('c', 12.05, 14.0),
        ])

        ll_b = ll_a

        assert ll_a is ll_b

    def test_iter(self):
        ll = LabelList(labels=[
            Label('some text'),
            Label('more text'),
            Label('text again'),
        ])

        labels = sorted(ll)

        assert labels[0].value == 'more text'
        assert labels[1].value == 'some text'
        assert labels[2].value == 'text again'

    def test_len(self):
        ll = LabelList(labels=[
            Label('some text'),
            Label('more text'),
            Label('text again'),
        ])

        assert len(ll) == 3

    def test_labels(self):
        ll = LabelList(labels=[
            Label('a'),
            Label('c'),
            Label('b'),
        ])

        labels = ll.labels
        assert isinstance(labels, list)
        assert sorted(labels) == [
            Label('a'),
            Label('b'),
            Label('c'),
        ]

    def test_start(self):
        ll = LabelList(idx='test', labels=[
            Label('c', 9.0, 12.0),
            Label('a', 0.2, 4.0),
            Label('b', 4.0, 8.0),
            Label('b', 7.9, 9.3),
            Label('c', 12.05, 14.0),
        ])

        assert ll.start == 0.2

    def test_end(self):
        ll = LabelList(idx='test', labels=[
            Label('c', 9.0, 12.0),
            Label('a', 0.2, 4.0),
            Label('b', 4.0, 8.0),
            Label('c', 12.05, 14.0),
            Label('b', 7.9, 9.3),
        ])

        assert ll.end == 14.0

    def test_end_returns_inf(self):
        ll = LabelList(idx='test', labels=[
            Label('c', 9.0, 12.0),
            Label('a', 0.2, 4.0),
            Label('b', 4.0, float('inf')),
            Label('c', 12.05, 14.0),
            Label('b', 7.9, 9.3),
        ])

        assert ll.end == float('inf')

    def test_add(self):
        ll = LabelList()

        assert len(ll) == 0

        label = Label('some text')
        ll.add(label)

        assert len(ll) == 1
        assert label.label_list == ll
        assert sorted(ll)[0] == label

    def test_addl(self):
        ll = LabelList()

        assert len(ll) == 0

        ll.addl('a', 12.3, 19.3)

        assert ll == LabelList(labels=[
            Label('a', 12.3, 19.3),
        ])

    def test_update(self):
        ll = LabelList()

        assert len(ll) == 0

        label_a = Label('some text')
        label_b = Label('more text')
        label_c = Label('text again')
        ll.update([label_a, label_b, label_c])

        assert len(ll) == 3
        assert label_a.label_list == ll
        assert label_b.label_list == ll
        assert label_c.label_list == ll

    def test_apply(self):
        ll = LabelList(labels=[
            Label('some text'),
            Label('more text'),
            Label('text again'),
        ])

        def apply_func(label):
            label.value = 'app {}'.format(label.value)

        ll.apply(apply_func)

        labels = sorted(ll)
        assert labels[0].value == 'app more text'
        assert labels[1].value == 'app some text'
        assert labels[2].value == 'app text again'

    def test_merge_overlaps(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('b', 4.0, 8.0),
            Label('b', 7.9, 9.3),
            Label('c', 9.0, 12.0),
            Label('c', 12.05, 14.0),
        ])
        ll.merge_overlaps()

        expected = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('b', 4.0, 9.3),
            Label('c', 9.0, 12.0),
            Label('c', 12.05, 14.0),
        ])

        assert ll == expected

    def test_merge_overlaps_with_threshold(self):
        ll = LabelList(idx='test', labels=[
            Label('c', 9.0, 12.0),
            Label('c', 12.05, 14.0),
        ])

        ll.merge_overlaps(threshold=0.1)

        expected = LabelList(idx='test', labels=[
            Label('c', 9.0, 14.0),
        ])

        assert ll == expected

    def test_merge_overlaps_with_multiple_consecutive_overlapping(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('b', 4.0, 8.0),
            Label('b', 7.9, 9.3),
            Label('b', 8.9, 10.9),
            Label('c', 9.0, 12.0)
        ])

        ll.merge_overlaps()

        expected = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('b', 4.0, 10.9),
            Label('c', 9.0, 12.0),
        ])

        assert ll == expected

    def test_merge_overlaps_with_multiple_overlapping(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('b', 4.0, 8.0),
            Label('b', 6.2, 9.3),
            Label('b', 7.1, 10.9),
            Label('c', 9.0, 12.0),
        ])

        ll.merge_overlaps()

        expected = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('b', 4.0, 10.9),
            Label('c', 9.0, 12.0),
        ])

        assert ll == expected

    def test_merge_overlaps_when_fully_contained(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('b', 4.0, 8.0),
            Label('b', 5.0, 7.3),
            Label('b', 7.9, 8.7),
            Label('c', 9.0, 12.0),
        ])

        ll.merge_overlaps()

        expected = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('b', 4.0, 8.7),
            Label('c', 9.0, 12.0),
        ])

        assert ll == expected

    def test_label_total_duration(self):
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

    def test_label_values(self):
        ll = LabelList(labels=[
            Label('a', 3.2, 4.5),
            Label('b', 5.1, 8.9),
            Label('c', 7.2, 10.5),
            Label('a', 10.5, 14),
            Label('c', 13, 14)
        ])

        assert ll.label_values() == ['a', 'b', 'c']

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

    def test_all_tokens(self):
        ll = LabelList(labels=[
            Label('some text'),
            Label('more text'),
            Label('text again'),
        ])

        assert sorted(ll.all_tokens()) == [
            'again',
            'more',
            'some',
            'text',
        ]

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
            Label('b', 4.5, float('inf')),
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
            Label('b x', 4.5, float('inf')),
            Label('c', 9.0, 12.0)
        ])

        with pytest.raises(ValueError):
            ll.tokenized()

    def test_separated(self):
        ll = LabelList(idx='a', labels=[
            Label('a', 3.2, 4.5),
            Label('b', 5.1, 8.9),
            Label('c', 7.2, 10.5),
            Label('a', 10.5, 14),
            Label('c', 13, 14)
        ])

        res = ll.separated()

        assert res['a'] == LabelList(idx='a', labels=[
            Label('a', 3.2, 4.5),
            Label('a', 10.5, 14)
        ])
        assert res['b'] == LabelList(idx='a', labels=[
            Label('b', 5.1, 8.9)
        ])
        assert res['c'] == LabelList(idx='a', labels=[
            Label('c', 7.2, 10.5),
            Label('c', 13, 14)
        ])

    def test_labels_in_range(self):
        ll = LabelList(labels=[
            Label('a', 3.2, 4.5),
            Label('b', 5.1, 8.9),
            Label('c', 7.2, 10.5),
            Label('a', 10.5, 14),
            Label('c', 13, 14)
        ])

        in_range = ll.labels_in_range(8.2, 12.5)

        assert sorted(in_range) == [
            Label('b', 5.1, 8.9),
            Label('c', 7.2, 10.5),
            Label('a', 10.5, 14)
        ]

    def test_labels_in_range_returns_only_fully_included(self):
        ll = LabelList(labels=[
            Label('a', 3.2, 4.5),
            Label('b', 5.1, 8.9),
            Label('c', 7.2, 10.5),
            Label('a', 10.5, 14),
            Label('c', 13, 15)
        ])

        in_range = ll.labels_in_range(7.2, 14.99, fully_included=True)

        assert sorted(in_range) == [
            Label('c', 7.2, 10.5),
            Label('a', 10.5, 14),
        ]

    def test_ranges(self):
        labels = [
            Label('a', 3.2, 4.5),
            Label('b', 5.1, 8.9),
            Label('c', 7.2, 10.5),
            Label('d', 10.5, 14)
        ]
        ll = LabelList(labels=labels)

        ranges = ll.ranges()

        r = next(ranges)
        assert 3.2 == r[0]
        assert 4.5 == r[1]
        assert labels[0] in r[2]

        r = next(ranges)
        assert 5.1 == r[0]
        assert 7.2 == r[1]
        assert labels[1] in r[2]

        r = next(ranges)
        assert 7.2 == r[0]
        assert 8.9 == r[1]
        assert labels[1] in r[2]
        assert labels[2] in r[2]

        r = next(ranges)
        assert 8.9 == r[0]
        assert 10.5 == r[1]
        assert labels[2] in r[2]

        r = next(ranges)
        assert 10.5 == r[0]
        assert 14 == r[1]
        assert labels[3] in r[2]

        with pytest.raises(StopIteration):
            next(ranges)

    def test_ranges_with_empty(self):
        labels = [
            Label('a', 3.2, 4.5),
            Label('b', 5.1, 8.9),
            Label('c', 7.2, 10.5),
            Label('d', 10.5, 14)
        ]
        ll = LabelList(labels=labels)

        ranges = ll.ranges(yield_ranges_without_labels=True)

        r = next(ranges)
        assert 3.2 == r[0]
        assert 4.5 == r[1]
        assert labels[0] in r[2]

        r = next(ranges)
        assert 4.5 == r[0]
        assert 5.1 == r[1]
        assert 0 == len(r[2])

        r = next(ranges)
        assert 5.1 == r[0]
        assert 7.2 == r[1]
        assert labels[1] in r[2]

        r = next(ranges)
        assert 7.2 == r[0]
        assert 8.9 == r[1]
        assert labels[1] in r[2]
        assert labels[2] in r[2]

        r = next(ranges)
        assert 8.9 == r[0]
        assert 10.5 == r[1]
        assert labels[2] in r[2]

        r = next(ranges)
        assert 10.5 == r[0]
        assert 14 == r[1]
        assert labels[3] in r[2]

        with pytest.raises(StopIteration):
            next(ranges)

    def test_ranges_include_labels(self):
        labels = [
            Label('a', 3.2, 4.5),
            Label('b', 5.1, 8.9)
        ]
        ll = LabelList(labels=labels)

        ranges = ll.ranges(include_labels=['a'])

        r = next(ranges)
        assert 3.2 == r[0]
        assert 4.5 == r[1]
        assert labels[0] in r[2]

        with pytest.raises(StopIteration):
            next(ranges)

    def test_ranges_zero_to_end(self):
        labels = [
            Label('a', 0, float('inf')),
            Label('b', 5.1, 8.9)
        ]
        ll = LabelList(labels=labels)

        ranges = ll.ranges()

        r = next(ranges)
        assert 0 == r[0]
        assert 5.1 == r[1]
        assert labels[0] in r[2]

        r = next(ranges)
        assert 5.1 == r[0]
        assert 8.9 == r[1]
        assert labels[0] in r[2]
        assert labels[1] in r[2]

        r = next(ranges)
        assert 8.9 == r[0]
        assert float('inf') == r[1]
        assert labels[0] in r[2]

        with pytest.raises(StopIteration):
            next(ranges)

    def test_ranges_with_same_start_times(self):
        labels = [
            Label('a', 1.2, 1.3),
            Label('b', 1.2, 5.6)
        ]
        ll = LabelList(labels=labels)

        ranges = ll.ranges()

        r = next(ranges)
        assert r[0] == 1.2
        assert r[1] == 1.3
        assert len(r[2]) == 2
        assert labels[0] in r[2]
        assert labels[1] in r[2]

        r = next(ranges)
        assert r[0] == 1.3
        assert r[1] == 5.6
        assert len(r[2]) == 1
        assert labels[1] in r[2]

        with pytest.raises(StopIteration):
            next(ranges)

    def test_split(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('b', 4.0, 8.0),
            Label('c', 9.0, 12.0)
        ])

        res = ll.split([1.9, 6.2, 10.5])

        assert res == [
            LabelList(idx='test', labels=[
                Label('a', 0.0, 1.9)
            ]),
            LabelList(idx='test', labels=[
                Label('a', 1.9, 4.0),
                Label('b', 4.0, 6.2),
            ]),
            LabelList(idx='test', labels=[
                Label('b', 6.2, 8.0),
                Label('c', 9.0, 10.5),
            ]),
            LabelList(idx='test', labels=[
                Label('c', 10.5, 12.0),
            ]),
        ]

    def test_split_unsorted_label_list(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('c', 9.0, 12.0),
            Label('b', 4.0, 8.0)
        ])

        res = ll.split([1.9, 6.2, 10.5])

        assert res == [
            LabelList(idx='test', labels=[
                Label('a', 0.0, 1.9),
            ]),
            LabelList(idx='test', labels=[
                Label('a', 1.9, 4.0),
                Label('b', 4.0, 6.2),
            ]),
            LabelList(idx='test', labels=[
                Label('b', 6.2, 8.0),
                Label('c', 9.0, 10.5),
            ]),
            LabelList(idx='test', labels=[
                Label('c', 10.5, 12.0),
            ]),
        ]

    def test_split_label_within_cutting_points_is_included(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('b', 4.0, 8.0),
            Label('c', 9.0, 12.0)
        ])

        res = ll.split([1.9, 10.5])

        assert len(res[1]) == 3
        assert sorted(res[1])[1].value == 'b'
        assert sorted(res[1])[1].start == 4.0
        assert sorted(res[1])[1].end == 8.0

    def test_split_with_endless_label(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('c', 4.0, float('inf'))
        ])

        res = ll.split([1.9, 10.5])

        assert len(res) == 3

        assert len(res[2]) == 1
        assert sorted(res[1])[1].value == 'c'
        assert sorted(res[1])[1].start == 4.0
        assert sorted(res[1])[1].end == 10.5
        assert sorted(res[2])[0].value == 'c'
        assert sorted(res[2])[0].start == 10.5
        assert sorted(res[2])[0].end == float('inf')

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
        assert sorted(res[0])[0].value == 'a'
        assert sorted(res[0])[0].start == 0.0
        assert sorted(res[0])[0].end == 4.2

        assert len(res[1]) == 2
        assert sorted(res[1])[0] == Label('a', 0.0, 4.8)
        assert sorted(res[1])[1] == Label('c', 4.8, 7.8)

    def test_split_first_label_not_splitted(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 9.0),
            Label('c', 9.0, 12.0)
        ])

        res = ll.split([11.2], shift_times=True)

        assert len(res) == 2

        assert len(res[0]) == 2
        assert sorted(res[0])[0] == Label('a', 0.0, 9.0)
        assert sorted(res[0])[1] == Label('c', 9.0, 11.2)

        assert len(res[1]) == 1
        assert sorted(res[1])[0] == Label('c', 0.0, pytest.approx(0.8))

    def test_split_single_label_that_doesnt_start_at_zero(self):
        ll = LabelList(idx='test', labels=[
            Label('c', 8.0, 12.0)
        ])

        res = ll.split([11.2], shift_times=True)

        assert len(res) == 2

        assert len(res[0]) == 1
        assert sorted(res[0])[0] == Label('c', 8.0, 11.2)

        assert len(res[1]) == 1
        assert sorted(res[1])[0] == Label('c', 0.0, pytest.approx(0.8))

    def test_split_with_overlap(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 0.0, 4.0),
            Label('b', 4.0, 8.0),
            Label('c', 9.0, 12.0)
        ])

        res = ll.split([1.9, 6.2, 10.5], overlap=2.0)

        assert res == [
            LabelList(idx='test', labels=[
                Label('a', 0.0, 3.9)
            ]),
            LabelList(idx='test', labels=[
                Label('a', 0, 4.0),
                Label('b', 4.0, 8.0)
            ]),
            LabelList(idx='test', labels=[
                Label('b', 4.2, 8.0),
                Label('c', 9.0, 12.0)]
            ),
            LabelList(idx='test', labels=[
                Label('c', 9.0, 12.0)
            ]),
        ]

    def test_split_with_overlap_label_only_touches_overlap_time(self):
        ll = LabelList(idx='test', labels=[
            Label('a', 2.0, 5.0),
            Label('b', 7.0, 8.0),
        ])

        res = ll.split([6.2, 10.5], overlap=2.0)

        assert res == [
            LabelList(idx='test', labels=[
                Label('a', 2.0, 5.0),
                Label('b', 7.0, 8.0),
            ]),

            LabelList(idx='test', labels=[
                Label('a', 4.2, 5.0),
                Label('b', 7.0, 8.0),
            ]),
            LabelList(idx='test', labels=[])
        ]

    def test_split_with_shift_and_overlap(self):
        ll = LabelList('test', labels=[
            Label('alpha', start=0.0, end=30.0),
            Label('bravo', start=20.0, end=42.0)
        ])

        res = ll.split([12.0, 24.0], shift_times=True, overlap=2.0)

        for x in res:
            print(x.labels)

        assert res == [
            LabelList(idx='test', labels=[
                Label('alpha', 0.0, 14.0),
            ]),

            LabelList(idx='test', labels=[
                Label('alpha', 0.0, 16.0),
                Label('bravo', 10.0, 16.0),
            ]),
            LabelList(idx='test', labels=[
                Label('alpha', 0.0, 8.0),
                Label('bravo', 0.0, 20.0),
            ])
        ]

    def test_create_single(self):
        ll = LabelList.create_single('bob')

        assert ll == LabelList(idx='default', labels=[
            Label('bob')
        ])

    def test_create_single_with_custom_idx(self):
        ll = LabelList.create_single('bob', idx='name')

        assert ll == LabelList(idx='name', labels=[
            Label('bob')
        ])

    def test_with_label_values(self):
        ll = LabelList.with_label_values([
            'a',
            'b',
            'c',
        ])

        assert ll == LabelList(labels=[
            Label('a'),
            Label('b'),
            Label('c'),
        ])

    def test_with_label_values_sets_correct_idx(self):
        ll = LabelList.with_label_values([
            'a',
            'b',
            'c',
        ], idx='letters')

        assert ll == LabelList(idx='letters', labels=[
            Label('a'),
            Label('b'),
            Label('c'),
        ])
