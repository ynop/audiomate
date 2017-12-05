import unittest

from pingu.corpus import assets


class CorpusTest(unittest.TestCase):
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
