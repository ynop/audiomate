import os.path
import unittest

import pytest

from pingu.corpus import assets
from pingu.corpus.utils import labellist
from pingu.corpus.utils.labellist import UnmappedLabelsException


class LabelMapperTest(unittest.TestCase):

    def test_relabel_maps_a_onto_b(self):
        label_list = assets.LabelList(labels=[
            assets.Label('a', 3.2, 4.5)
        ])

        actual = labellist.relabel(label_list, {('a',): 'b'})

        assert len(actual) == 1
        assert actual[0].start == 3.2
        assert actual[0].end == 4.5
        assert actual[0].value == 'b'

    def test_relabel_flattens_partial_overlap_into_combined_label(self):
        projections = {
            ('a',): 'a',
            ('b',): 'b',
            ('c',): 'c',
            ('a', 'b',): 'a_b',
            ('a', 'b', 'c',): 'a_b_c',
            ('b', 'c',): 'b_c',
        }

        label_list = assets.LabelList(labels=[
            assets.Label('a', 3.2, 4.5),
            assets.Label('b', 4.0, 4.9),
            assets.Label('c', 4.2, 5.1)
        ])

        actual = labellist.relabel(label_list, projections)

        assert len(actual) == 5

        assert actual[0].start == 3.2
        assert actual[0].end == 4.0
        assert actual[0].value == 'a'

        assert actual[1].start == 4.0
        assert actual[1].end == 4.2
        assert actual[1].value == 'a_b'

        assert actual[2].start == 4.2
        assert actual[2].end == 4.5
        assert actual[2].value == 'a_b_c'

        assert actual[3].start == 4.5
        assert actual[3].end == 4.9
        assert actual[3].value == 'b_c'

        assert actual[4].start == 4.9
        assert actual[4].end == 5.1
        assert actual[4].value == 'c'

    def test_relabel_flattens_full_overlap_into_combined_label(self):
        projections = {
            ('a',): 'a',
            ('b',): 'b',
            ('a', 'b'): 'a_b',
        }

        label_list = assets.LabelList(labels=[
            assets.Label('a', 3.2, 4.9),
            assets.Label('b', 3.9, 4.5)
        ])

        actual = labellist.relabel(label_list, projections)

        assert len(actual) == 3

        assert actual[0].start == 3.2
        assert actual[0].end == 3.9
        assert actual[0].value == 'a'

        assert actual[1].start == 3.9
        assert actual[1].end == 4.5
        assert actual[1].value == 'a_b'

        assert actual[2].start == 4.5
        assert actual[2].end == 4.9
        assert actual[2].value == 'a'

    def test_relabel_removes_unwanted_labels(self):
        projections = {
            ('a',): '',
            ('b',): 'b',
        }

        label_list = assets.LabelList(labels=[
            assets.Label('a', 3.2, 4.4),
            assets.Label('b', 4.4, 5.1)
        ])

        actual = labellist.relabel(label_list, projections)

        assert len(actual) == 1

        assert actual[0].start == 4.4
        assert actual[0].end == 5.1
        assert actual[0].value == 'b'

    def test_relabel_removes_overlapping_segment(self):
        projections = {
            ('a',): 'a',
            ('a', 'b',): '',
            ('b',): 'b',
        }

        label_list = assets.LabelList(labels=[
            assets.Label('a', 3.2, 5.1),
            assets.Label('b', 4.2, 4.7)
        ])

        actual = labellist.relabel(label_list, projections)

        assert len(actual) == 2

        assert actual[0].start == 3.2
        assert actual[0].end == 4.2
        assert actual[0].value == 'a'

        assert actual[1].start == 4.7
        assert actual[1].end == 5.1
        assert actual[1].value == 'a'

    def test_relabel_throws_error_if_unmapped_labels_are_detected(self):
        label_list = assets.LabelList(labels=[
            assets.Label('a', 3.2, 5.1),
            assets.Label('b', 4.2, 4.7),
            assets.Label('c', 4.3, 4.8)
        ])

        unmapped_combinations = [('a', 'b'), ('a', 'b', 'c'), ('a', 'c')]
        expected_message = 'Unmapped combinations: {}'.format(unmapped_combinations)

        with pytest.raises(UnmappedLabelsException) as ex:
            labellist.relabel(label_list, {('a',): 'foo'})

        assert ex.value.message == expected_message

    def test_load_projections_from_file(self):
        path = os.path.join(os.path.dirname(__file__), 'projections.txt')
        projections = labellist.load_projections(path)

        assert len(projections) == 3

        assert ('b',) in projections
        assert projections[('b',)] == 'foo'

        assert ('a', 'b',) in projections
        assert projections[('a', 'b',)] == 'a_b'

        assert ('a',) in projections
        assert projections[('a',)] == 'bar'

    def test_all_projections_missing_if_no_projections_defined(self):
        label_list = assets.LabelList(labels=[
            assets.Label('b', 3.2, 4.5),
            assets.Label('a', 4.0, 4.9),
            assets.Label('c', 4.2, 5.1)
        ])

        unmapped_combinations = labellist.find_missing_projections(label_list, {})

        assert len(unmapped_combinations) == 5
        assert ('b',) in unmapped_combinations
        assert ('a', 'b',) in unmapped_combinations
        assert ('a', 'b', 'c',) in unmapped_combinations
        assert ('a', 'c',) in unmapped_combinations
        assert ('c',) in unmapped_combinations

    def test_all_missing_projections_found(self):
        projections = {
            ('a', 'b',): 'foo',
            ('c',): 'bar'
        }

        label_list = assets.LabelList(labels=[
            assets.Label('b', 3.2, 4.5),
            assets.Label('a', 4.0, 4.9),
            assets.Label('c', 4.2, 5.1)
        ])

        unmapped_combinations = labellist.find_missing_projections(label_list, projections)

        assert len(unmapped_combinations) == 3
        assert ('b',) in unmapped_combinations
        assert ('a', 'b', 'c',) in unmapped_combinations
        assert ('a', 'c',) in unmapped_combinations

    def test_no_missing_projections_if_projection_complete(self):
        projections = {
            ('b',): 'foo',
            ('a', 'b',): 'foo',
            ('a', 'b', 'c',): 'foo',
            ('a', 'c',): 'foo',
            ('c',): 'bar'
        }

        label_list = assets.LabelList(labels=[
            assets.Label('b', 3.2, 4.5),
            assets.Label('a', 4.0, 4.9),
            assets.Label('c', 4.2, 5.1)
        ])

        unmapped_combinations = labellist.find_missing_projections(label_list, projections)

        assert len(unmapped_combinations) == 0
