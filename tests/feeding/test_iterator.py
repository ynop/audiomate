import os

import numpy as np

from audiomate import containers
from audiomate.corpus import subset
from audiomate import feeding
from audiomate.feeding import iterator

import pytest
from tests import resources


@pytest.fixture
def sample_partition_data():
    info = feeding.PartitionInfo()
    info.utt_ids = ['utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5', ]

    data = feeding.PartitionData(info)
    data.utt_data = [
        [np.arange(20).reshape(5, 4), np.arange(20).reshape(5, 4) + 10],
        [np.arange(28).reshape(7, 4), np.arange(28).reshape(7, 4) + 10],
        [np.arange(36).reshape(9, 4), np.arange(36).reshape(9, 4) + 10],
        [np.arange(8).reshape(2, 4), np.arange(8).reshape(2, 4) + 10],
        [np.arange(16).reshape(4, 4), np.arange(16).reshape(4, 4) + 10]
    ]

    return data


class TestDataIterator:

    def test_init_with_utterance_list(self):
        it = feeding.DataIterator(['utt-1', 'utt-2'], [containers.Container('blub')])
        assert set(it.utt_ids) == {'utt-1', 'utt-2'}

    def test_init_with_corpus(self):
        corpus = resources.create_dataset()
        it = feeding.DataIterator(corpus, [containers.Container('blub')])
        assert set(it.utt_ids) == set(corpus.utterances.keys())

    def test_init_with_corpus_view(self):
        corpus = resources.create_dataset()
        subview = subset.Subview(corpus, filter_criteria=[
            subset.MatchingUtteranceIdxFilter(utterance_idxs={'utt-1', 'utt-2', 'utt-4'})
        ])

        it = feeding.DataIterator(subview, [containers.Container('blub')])
        assert set(it.utt_ids) == set(subview.utterances.keys())

    def test_init_throws_error_when_no_container_is_given(self):
        corpus = resources.create_dataset()

        with pytest.raises(ValueError):
            feeding.DataIterator(corpus, [])


class TestMultiFramePartitionData:

    def test_get_utt_regions(self, sample_partition_data):
        frame_data = iterator.MultiFramePartitionData(sample_partition_data, 3, shuffle=False)
        regions = frame_data.get_utt_regions()

        assert regions[0][0] == 0
        assert regions[0][1] == 2

        assert regions[1][0] == 2
        assert regions[1][1] == 3

        assert regions[2][0] == 5
        assert regions[2][1] == 3

        assert regions[3][0] == 8
        assert regions[3][1] == 1

        assert regions[4][0] == 9
        assert regions[4][1] == 2

    def test_get_length(self, sample_partition_data):
        frame_data = iterator.MultiFramePartitionData(sample_partition_data, 3, shuffle=False)
        assert len(frame_data) == 11

    def test_get_item_at_start_of_utterance(self, sample_partition_data):
        frame_data = iterator.MultiFramePartitionData(sample_partition_data, 3, shuffle=False)

        assert len(frame_data[2]) == 2
        assert np.array_equal(frame_data[2][0], np.arange(12).reshape(3, 4))
        assert np.array_equal(frame_data[2][1], np.arange(12).reshape(3, 4) + 10)

    def test_get_item_in_middle_of_utterance(self, sample_partition_data):
        frame_data = iterator.MultiFramePartitionData(sample_partition_data, 3, shuffle=False)

        assert len(frame_data[6]) == 2
        assert np.array_equal(frame_data[6][0], np.arange(12).reshape(3, 4) + 12)
        assert np.array_equal(frame_data[6][1], np.arange(12).reshape(3, 4) + 22)

    def test_get_item_at_end_of_utterance(self, sample_partition_data):
        frame_data = iterator.MultiFramePartitionData(sample_partition_data, 3, shuffle=False)

        assert len(frame_data[8]) == 2
        assert np.array_equal(frame_data[8][0], np.arange(8).reshape(2, 4))
        assert np.array_equal(frame_data[8][1], np.arange(8).reshape(2, 4) + 10)


class TestMultiFrameIterator(object):

    def test_next_emits_no_frames_if_file_is_empty(self, tmpdir):
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        cont = containers.Container(file_path)
        cont.open()

        frames = tuple(iterator.MultiFrameIterator([], [cont], '120', 5))
        assert 0 == len(frames)

    def test_next_emits_no_features_if_data_set_is_empty(self, tmpdir):
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        cont = containers.Container(file_path)
        cont.open()
        cont.set('utt-1', np.array([]))

        frames = tuple(iterator.MultiFrameIterator(['utt-1'], [cont], '120', 5))
        assert 0 == len(frames)

    def test_next_emits_all_features_in_sequential_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        cont = containers.Container(file_path)
        cont.open()
        cont.set('utt-1', ds1)
        cont.set('utt-2', ds2)

        frames = tuple(iterator.MultiFrameIterator(['utt-1', 'utt-2'], [cont], '120', 5, shuffle=False))
        assert 2 == len(frames)

        assert np.allclose(([[0.1, 0.1, 0.1, 0.1, 0.1],
                             [0.2, 0.2, 0.2, 0.2, 0.2]]), frames[0][0])
        assert np.allclose(([[0.3, 0.3, 0.3, 0.3, 0.3],
                             [0.4, 0.4, 0.4, 0.4, 0.4],
                             [0.5, 0.5, 0.5, 0.5, 0.5]]), frames[1][0])

    def test_next_emits_all_features_in_random_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        cont = containers.Container(file_path)
        cont.open()
        cont.set('utt-1', ds1)
        cont.set('utt-2', ds2)
        cont.set('utt-3', ds3)

        frames = tuple(iterator.MultiFrameIterator(['utt-1', 'utt-2', 'utt-3'], [cont], 120, 2, shuffle=True, seed=6))

        assert 4 == len(frames)

        assert np.allclose(([[0.5, 0.5, 0.5, 0.5, 0.5]]), frames[0][0])
        assert np.allclose(([[0.3, 0.3, 0.3, 0.3, 0.3],
                             [0.4, 0.4, 0.4, 0.4, 0.4]]), frames[1][0])
        assert np.allclose(([[0.6, 0.6, 0.6, 0.6, 0.6]]), frames[2][0])
        assert np.allclose(([[0.1, 0.1, 0.1, 0.1, 0.1],
                             [0.2, 0.2, 0.2, 0.2, 0.2]]), frames[3][0])

    def test_next_emits_features_only_from_included_ds_in_sequential_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        cont = containers.Container(file_path)
        cont.open()
        cont.set('utt-1', ds1)
        cont.set('utt-2', ds2)
        cont.set('utt-3', ds3)

        frames = tuple(iterator.MultiFrameIterator(['utt-1', 'utt-3'], [cont], 120, 2, shuffle=False))

        assert 2 == len(frames)
        assert np.allclose(([[0.1, 0.1, 0.1, 0.1, 0.1],
                             [0.2, 0.2, 0.2, 0.2, 0.2]]), frames[0][0])
        assert np.allclose(([[0.6, 0.6, 0.6, 0.6, 0.6],
                             [0.7, 0.7, 0.7, 0.7, 0.7]]), frames[1][0])

    def test_next_emits_features_only_from_included_ds_in_random_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        cont = containers.Container(file_path)
        cont.open()
        cont.set('utt-1', ds1)
        cont.set('utt-2', ds2)
        cont.set('utt-3', ds3)

        frames = tuple(iterator.MultiFrameIterator(['utt-1', 'utt-3'], [cont], 120, 2, shuffle=True, seed=1))

        assert 2 == len(frames)

        assert np.allclose(([[0.6, 0.6, 0.6, 0.6, 0.6],
                             [0.7, 0.7, 0.7, 0.7, 0.7]]), frames[0][0])
        assert np.allclose(([[0.1, 0.1, 0.1, 0.1, 0.1],
                             [0.2, 0.2, 0.2, 0.2, 0.2]]), frames[1][0])

    def test_next_emits_all_features_if_partition_spans_multiple_data_sets_in_sequential_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        cont = containers.Container(file_path)
        cont.open()
        cont.set('utt-1', ds1)
        cont.set('utt-2', ds2)
        cont.set('utt-3', ds3)

        frames = tuple(iterator.MultiFrameIterator(['utt-1', 'utt-2', 'utt-3'], [cont], 240, 2, shuffle=False))

        assert 4 == len(frames)

        assert np.allclose(([[0.1, 0.1, 0.1, 0.1, 0.1],
                             [0.2, 0.2, 0.2, 0.2, 0.2]]), frames[0][0])
        assert np.allclose(([[0.3, 0.3, 0.3, 0.3, 0.3],
                             [0.4, 0.4, 0.4, 0.4, 0.4]]), frames[1][0])
        assert np.allclose(([[0.5, 0.5, 0.5, 0.5, 0.5]]), frames[2][0])
        assert np.allclose(([[0.6, 0.6, 0.6, 0.6, 0.6],
                             [0.7, 0.7, 0.7, 0.7, 0.7]]), frames[3][0])

    def test_next_emits_all_features_if_partition_spans_multiple_data_sets_in_random_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        cont = containers.Container(file_path)
        cont.open()
        cont.set('utt-1', ds1)
        cont.set('utt-2', ds2)
        cont.set('utt-3', ds3)

        frames = tuple(iterator.MultiFrameIterator(['utt-1', 'utt-2', 'utt-3'], [cont], 240, 2, shuffle=True, seed=12))

        assert 4 == len(frames)

        assert np.allclose(([[0.5, 0.5, 0.5, 0.5, 0.5]]), frames[0][0])
        assert np.allclose(([[0.3, 0.3, 0.3, 0.3, 0.3],
                             [0.4, 0.4, 0.4, 0.4, 0.4]]), frames[1][0])
        assert np.allclose(([[0.1, 0.1, 0.1, 0.1, 0.1],
                             [0.2, 0.2, 0.2, 0.2, 0.2]]), frames[2][0])
        assert np.allclose(([[0.6, 0.6, 0.6, 0.6, 0.6],
                             [0.7, 0.7, 0.7, 0.7, 0.7]]), frames[3][0])

    def test_next_emits_chunks_with_length(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        cont = containers.Container(file_path)
        cont.open()
        cont.set('utt-1', ds1)
        cont.set('utt-2', ds2)
        cont.set('utt-3', ds3)

        frames = tuple(iterator.MultiFrameIterator(['utt-1', 'utt-2', 'utt-3'], [cont], 120, 2, return_length=True,
                                                   shuffle=True, seed=6))

        assert 4 == len(frames)

        assert np.allclose(([[0.5, 0.5, 0.5, 0.5, 0.5]]), frames[0][0])
        assert np.allclose(([[0.3, 0.3, 0.3, 0.3, 0.3],
                             [0.4, 0.4, 0.4, 0.4, 0.4]]), frames[1][0])
        assert np.allclose(([[0.6, 0.6, 0.6, 0.6, 0.6]]), frames[2][0])
        assert np.allclose(([[0.1, 0.1, 0.1, 0.1, 0.1],
                             [0.2, 0.2, 0.2, 0.2, 0.2]]), frames[3][0])

        assert frames[0][1] == 1
        assert frames[1][1] == 2
        assert frames[2][1] == 1
        assert frames[3][1] == 2

    def test_next_emits_chunks_with_padding(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        cont = containers.Container(file_path)
        cont.open()
        cont.set('utt-1', ds1)
        cont.set('utt-2', ds2)
        cont.set('utt-3', ds3)

        frames = tuple(iterator.MultiFrameIterator(['utt-1', 'utt-2', 'utt-3'], [cont], 120, 2, pad=True,
                                                   shuffle=True, seed=6))

        assert 4 == len(frames)

        assert np.allclose(([[0.5, 0.5, 0.5, 0.5, 0.5],
                             [0.0, 0.0, 0.0, 0.0, 0.0]]), frames[0][0])
        assert np.allclose(([[0.3, 0.3, 0.3, 0.3, 0.3],
                             [0.4, 0.4, 0.4, 0.4, 0.4]]), frames[1][0])
        assert np.allclose(([[0.6, 0.6, 0.6, 0.6, 0.6],
                             [0.0, 0.0, 0.0, 0.0, 0.0]]), frames[2][0])
        assert np.allclose(([[0.1, 0.1, 0.1, 0.1, 0.1],
                             [0.2, 0.2, 0.2, 0.2, 0.2]]), frames[3][0])

        assert frames[0][1] == 1
        assert frames[1][1] == 2
        assert frames[2][1] == 1
        assert frames[3][1] == 2


class TestFrameIterator(object):

    def test_next_emits_no_frames_if_file_is_empty(self, tmpdir):
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        cont = containers.Container(file_path)
        cont.open()

        frames = tuple(iterator.FrameIterator([], [cont], 120))
        assert 0 == len(frames)

    def test_next_emits_no_features_if_data_set_is_empty(self, tmpdir):
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        cont = containers.Container(file_path)
        cont.open()
        cont.set('utt-1', np.array([]))

        frames = tuple(iterator.FrameIterator(['utt-1'], [cont], 120))
        assert 0 == len(frames)

    def test_next_emits_all_features_in_sequential_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        cont = containers.Container(file_path)
        cont.open()
        cont.set('utt-1', ds1)
        cont.set('utt-2', ds2)

        frames = tuple(iterator.FrameIterator(['utt-1', 'utt-2'], [cont], 120, shuffle=False))
        assert 5 == len(frames)

        assert np.allclose(([0.1, 0.1, 0.1, 0.1, 0.1]), frames[0][0])
        assert np.allclose(([0.2, 0.2, 0.2, 0.2, 0.2]), frames[1][0])
        assert np.allclose(([0.3, 0.3, 0.3, 0.3, 0.3]), frames[2][0])
        assert np.allclose(([0.4, 0.4, 0.4, 0.4, 0.4]), frames[3][0])
        assert np.allclose(([0.5, 0.5, 0.5, 0.5, 0.5]), frames[4][0])

    def test_next_emits_all_features_in_random_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        cont = containers.Container(file_path)
        cont.open()
        cont.set('utt-1', ds1)
        cont.set('utt-2', ds2)
        cont.set('utt-3', ds3)

        frames = tuple(iterator.FrameIterator(['utt-1', 'utt-2', 'utt-3'], [cont], 120, shuffle=True, seed=136))

        assert 6 == len(frames)

        assert np.allclose(([0.6, 0.6, 0.6, 0.6, 0.6]), frames[0][0])
        assert np.allclose(([0.1, 0.1, 0.1, 0.1, 0.1]), frames[1][0])
        assert np.allclose(([0.2, 0.2, 0.2, 0.2, 0.2]), frames[2][0])
        assert np.allclose(([0.3, 0.3, 0.3, 0.3, 0.3]), frames[3][0])
        assert np.allclose(([0.5, 0.5, 0.5, 0.5, 0.5]), frames[4][0])
        assert np.allclose(([0.4, 0.4, 0.4, 0.4, 0.4]), frames[5][0])

    def test_next_emits_features_only_from_included_ds_in_sequential_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        cont = containers.Container(file_path)
        cont.open()
        cont.set('utt-1', ds1)
        cont.set('utt-2', ds2)
        cont.set('utt-3', ds3)

        frames = tuple(iterator.FrameIterator(['utt-1', 'utt-3'], [cont], 120, shuffle=False))

        assert 4 == len(frames)

        assert np.allclose(([0.1, 0.1, 0.1, 0.1, 0.1]), frames[0][0])
        assert np.allclose(([0.2, 0.2, 0.2, 0.2, 0.2]), frames[1][0])
        assert np.allclose(([0.6, 0.6, 0.6, 0.6, 0.6]), frames[2][0])
        assert np.allclose(([0.7, 0.7, 0.7, 0.7, 0.7]), frames[3][0])

    def test_next_emits_features_only_from_included_ds_in_random_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        cont = containers.Container(file_path)
        cont.open()
        cont.set('utt-1', ds1)
        cont.set('utt-2', ds2)
        cont.set('utt-3', ds3)

        frames = tuple(iterator.FrameIterator(['utt-1', 'utt-3'], [cont], 120, shuffle=True, seed=1236))

        assert 4 == len(frames)

        assert np.allclose(([0.1, 0.1, 0.1, 0.1, 0.1]), frames[0][0])
        assert np.allclose(([0.2, 0.2, 0.2, 0.2, 0.2]), frames[1][0])
        assert np.allclose(([0.7, 0.7, 0.7, 0.7, 0.7]), frames[2][0])
        assert np.allclose(([0.6, 0.6, 0.6, 0.6, 0.6]), frames[3][0])

    def test_next_emits_all_features_if_partition_spans_multiple_data_sets_in_sequential_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        cont = containers.Container(file_path)
        cont.open()
        cont.set('utt-1', ds1)
        cont.set('utt-2', ds2)
        cont.set('utt-3', ds3)

        frames = tuple(iterator.FrameIterator(['utt-1', 'utt-2', 'utt-3'], [cont], 240, shuffle=False))

        assert 7 == len(frames)

        assert np.allclose(([0.1, 0.1, 0.1, 0.1, 0.1]), frames[0][0])
        assert np.allclose(([0.2, 0.2, 0.2, 0.2, 0.2]), frames[1][0])
        assert np.allclose(([0.3, 0.3, 0.3, 0.3, 0.3]), frames[2][0])
        assert np.allclose(([0.4, 0.4, 0.4, 0.4, 0.4]), frames[3][0])
        assert np.allclose(([0.5, 0.5, 0.5, 0.5, 0.5]), frames[4][0])
        assert np.allclose(([0.6, 0.6, 0.6, 0.6, 0.6]), frames[5][0])
        assert np.allclose(([0.7, 0.7, 0.7, 0.7, 0.7]), frames[6][0])

    def test_next_emits_all_features_if_partition_spans_multiple_data_sets_in_random_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        cont = containers.Container(file_path)
        cont.open()
        cont.set('utt-1', ds1)
        cont.set('utt-2', ds2)
        cont.set('utt-3', ds3)

        frames = tuple(iterator.FrameIterator(['utt-1', 'utt-2', 'utt-3'], [cont], 240, shuffle=True, seed=333))

        assert 7 == len(frames)

        assert np.allclose(([0.5, 0.5, 0.5, 0.5, 0.5]), frames[0][0])
        assert np.allclose(([0.3, 0.3, 0.3, 0.3, 0.3]), frames[1][0])
        assert np.allclose(([0.4, 0.4, 0.4, 0.4, 0.4]), frames[2][0])
        assert np.allclose(([0.2, 0.2, 0.2, 0.2, 0.2]), frames[3][0])
        assert np.allclose(([0.1, 0.1, 0.1, 0.1, 0.1]), frames[4][0])
        assert np.allclose(([0.6, 0.6, 0.6, 0.6, 0.6]), frames[5][0])
        assert np.allclose(([0.7, 0.7, 0.7, 0.7, 0.7]), frames[6][0])
