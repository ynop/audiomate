import os.path

import pytest

from audiomate.corpus import assets
from audiomate.corpus.utils import relabeling


class TestLabelListUtilities(object):

    def test_relabel_maps_a_onto_b(self):
        label_list = assets.LabelList(labels=[
            assets.Label('a', 3.2, 4.5)
        ])

        actual = relabeling.relabel(label_list, {('a',): 'b'})

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

        actual = relabeling.relabel(label_list, projections)

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

        actual = relabeling.relabel(label_list, projections)

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

        actual = relabeling.relabel(label_list, projections)

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

        actual = relabeling.relabel(label_list, projections)

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

        with pytest.raises(relabeling.UnmappedLabelsException) as ex:
            relabeling.relabel(label_list, {('a',): 'foo'})

        assert ex.value.message == expected_message

    def test_relabel_proceeds_despite_unmapped_labels_in_presence_of_wildcard_rule(self):
        label_list = assets.LabelList(labels=[
            assets.Label('a', 3.2, 5.1),
            assets.Label('b', 4.2, 4.7),
            assets.Label('c', 4.3, 4.8)
        ])

        actual = relabeling.relabel(label_list, {('a',): 'new_label_a', ('**',): 'catch_all'})

        assert len(actual) == 5

        assert actual[0].start == 3.2
        assert actual[0].end == 4.2
        assert actual[0].value == 'new_label_a'

        assert actual[1].start == 4.2
        assert actual[1].end == 4.3
        assert actual[1].value == 'catch_all'

        assert actual[2].start == 4.3
        assert actual[2].end == 4.7
        assert actual[2].value == 'catch_all'

        assert actual[3].start == 4.7
        assert actual[3].end == 4.8
        assert actual[3].value == 'catch_all'

        assert actual[4].start == 4.8
        assert actual[4].end == 5.1
        assert actual[4].value == 'new_label_a'

    def test_load_projections_from_file(self):
        path = os.path.join(os.path.dirname(__file__), 'projections.txt')
        projections = relabeling.load_projections(path)

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

        unmapped_combinations = relabeling.find_missing_projections(label_list, {})

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

        unmapped_combinations = relabeling.find_missing_projections(label_list, projections)

        assert len(unmapped_combinations) == 3
        assert ('b',) in unmapped_combinations
        assert ('a', 'b', 'c',) in unmapped_combinations
        assert ('a', 'c',) in unmapped_combinations

    def test_no_duplicate_missing_projections_reported(self):
        label_list = assets.LabelList(labels=[
            assets.Label('b', 1.0, 2.0),
            assets.Label('a', 1.5, 2.5),
            assets.Label('b', 3.0, 4.0),
            assets.Label('a', 3.5, 4.5),
        ])

        unmapped_combinations = relabeling.find_missing_projections(label_list, {})

        assert len(unmapped_combinations) == 3
        assert ('b',) in unmapped_combinations
        assert ('a', 'b',) in unmapped_combinations
        assert ('a',) in unmapped_combinations

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

        unmapped_combinations = relabeling.find_missing_projections(label_list, projections)

        assert len(unmapped_combinations) == 0

    def test_no_missing_projections_if_covered_by_catch_all_rule(self):
        projections = {
            ('b',): 'new_label_b',
            ('**',): 'new_label_all',
        }

        label_list = assets.LabelList(labels=[
            assets.Label('b', 3.2, 4.5),
            assets.Label('a', 4.0, 4.9),
            assets.Label('c', 4.2, 5.1)
        ])

        unmapped_combinations = relabeling.find_missing_projections(label_list, projections)

        assert len(unmapped_combinations) == 0

    def test_missing_projections_are_naturally_sorted(self):
        label_list = assets.LabelList(labels=[
            assets.Label('b', 1.0, 2.0),
            assets.Label('a', 1.5, 2.5),
        ])

        unmapped_combinations = relabeling.find_missing_projections(label_list, {})

        assert len(unmapped_combinations) == 3
        assert unmapped_combinations[0] == ('a',)
        assert unmapped_combinations[1] == ('a', 'b',)
        assert unmapped_combinations[2] == ('b',)
