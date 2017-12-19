import unittest

from pingu.corpus import assets


class TestLabelList(unittest.TestCase):
    def test_append(self):
        ll = assets.LabelList()

        label = assets.Label('some text')
        ll.append(label)

        assert len(ll) == 1
        assert label.label_list == ll

    def test_extend(self):
        ll = assets.LabelList()

        label_a = assets.Label('some text')
        label_b = assets.Label('more text')
        label_c = assets.Label('text again')
        ll.extend([label_a, label_b, label_c])

        assert len(ll) == 3
        assert label_a.label_list == ll
        assert label_b.label_list == ll
        assert label_c.label_list == ll

    def test_ranges(self):
        ll = assets.LabelList(labels=[
            assets.Label('a', 3.2, 4.5),
            assets.Label('b', 5.1, 8.9),
            assets.Label('c', 7.2, 10.5),
            assets.Label('d', 10.5, 14)
        ])

        ranges = ll.ranges()

        r = next(ranges)
        self.assertEqual(3.2, r[0])
        self.assertEqual(4.5, r[1])
        self.assertIn(ll[0], r[2])

        r = next(ranges)
        self.assertEqual(5.1, r[0])
        self.assertEqual(7.2, r[1])
        self.assertIn(ll[1], r[2])

        r = next(ranges)
        self.assertEqual(7.2, r[0])
        self.assertEqual(8.9, r[1])
        self.assertIn(ll[1], r[2])
        self.assertIn(ll[2], r[2])

        r = next(ranges)
        self.assertEqual(8.9, r[0])
        self.assertEqual(10.5, r[1])
        self.assertIn(ll[2], r[2])

        r = next(ranges)
        self.assertEqual(10.5, r[0])
        self.assertEqual(14, r[1])
        self.assertIn(ll[3], r[2])

        with self.assertRaises(StopIteration):
            next(ranges)

    def test_ranges_with_empty(self):
        ll = assets.LabelList(labels=[
            assets.Label('a', 3.2, 4.5),
            assets.Label('b', 5.1, 8.9),
            assets.Label('c', 7.2, 10.5),
            assets.Label('d', 10.5, 14)
        ])

        ranges = ll.ranges(yield_ranges_without_labels=True)

        r = next(ranges)
        self.assertEqual(3.2, r[0])
        self.assertEqual(4.5, r[1])
        self.assertIn(ll[0], r[2])

        r = next(ranges)
        self.assertEqual(4.5, r[0])
        self.assertEqual(5.1, r[1])
        self.assertEqual(0, len(r[2]))

        r = next(ranges)
        self.assertEqual(5.1, r[0])
        self.assertEqual(7.2, r[1])
        self.assertIn(ll[1], r[2])

        r = next(ranges)
        self.assertEqual(7.2, r[0])
        self.assertEqual(8.9, r[1])
        self.assertIn(ll[1], r[2])
        self.assertIn(ll[2], r[2])

        r = next(ranges)
        self.assertEqual(8.9, r[0])
        self.assertEqual(10.5, r[1])
        self.assertIn(ll[2], r[2])

        r = next(ranges)
        self.assertEqual(10.5, r[0])
        self.assertEqual(14, r[1])
        self.assertIn(ll[3], r[2])

        with self.assertRaises(StopIteration):
            next(ranges)

    def test_ranges_include_labels(self):
        ll = assets.LabelList(labels=[
            assets.Label('a', 3.2, 4.5),
            assets.Label('b', 5.1, 8.9)
        ])

        ranges = ll.ranges(include_labels=['a'])

        r = next(ranges)
        self.assertEqual(3.2, r[0])
        self.assertEqual(4.5, r[1])
        self.assertIn(ll[0], r[2])

        with self.assertRaises(StopIteration):
            next(ranges)

    def test_ranges_zero_to_end(self):
        ll = assets.LabelList(labels=[
            assets.Label('a', 0, -1),
            assets.Label('b', 5.1, 8.9)
        ])

        ranges = ll.ranges()

        r = next(ranges)
        self.assertEqual(0, r[0])
        self.assertEqual(5.1, r[1])
        self.assertIn(ll[0], r[2])

        r = next(ranges)
        self.assertEqual(5.1, r[0])
        self.assertEqual(8.9, r[1])
        self.assertIn(ll[0], r[2])
        self.assertIn(ll[1], r[2])

        r = next(ranges)
        self.assertEqual(8.9, r[0])
        self.assertEqual(-1, r[1])
        self.assertIn(ll[0], r[2])

        with self.assertRaises(StopIteration):
            next(ranges)

    def test_ranges_with_same_start_times(self):
        ll = assets.LabelList(labels=[
            assets.Label('a', 1.2, 1.3),
            assets.Label('b', 1.2, 5.6)
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

        with self.assertRaises(StopIteration):
            next(ranges)

    def test_label_count(self):
        ll = assets.LabelList(labels=[
            assets.Label('a', 3.2, 4.5),
            assets.Label('b', 5.1, 8.9),
            assets.Label('c', 7.2, 10.5),
            assets.Label('a', 10.5, 14),
            assets.Label('c', 13, 14)
        ])

        res = ll.label_count()

        self.assertEqual(2, res['a'])
        self.assertEqual(1, res['b'])
        self.assertEqual(2, res['c'])


class TestLabel(object):
    def test_lt_start_time_considered_first(self):
        a = assets.Label('some label', 1.0, 2.0)
        b = assets.Label('some label', 1.1, 2.0)

        assert a < b

    def test_lt_end_time_considered_second(self):
        a = assets.Label('some label', 1.0, 1.9)
        b = assets.Label('some label', 1.0, 2.0)

        assert a < b

    def test_lt_end_time_properly_handles_custom_infinity(self):
        a = assets.Label('some label', 1.0, 1.9)
        b = assets.Label('some label', 1.0, -1)

        assert a < b

    def test_lt_value_considered_third(self):
        a = assets.Label('some label a', 1.0, 2.0)
        b = assets.Label('some label b', 1.0, 2.0)

        assert a < b

    def test_lt_value_ignores_capitalization(self):
        a = assets.Label('some label A', 1.0, 2.0)
        b = assets.Label('some label a', 1.0, 2.0)

        assert not a < b  # not == because == tests different method
        assert not a > b  # not == because == tests different method

    def test_eq_ignores_capitalization(self):
        a = assets.Label('some label A', 1.0, 2.0)
        b = assets.Label('some label a', 1.0, 2.0)

        assert a == b
