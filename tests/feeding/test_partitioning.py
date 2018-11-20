import os

import numpy as np
import h5py

from audiomate.feeding import PartitioningFeatureIterator
from audiomate.feeding import partitioning
from audiomate import containers

import pytest


class TestPartitioningContainerLoader:

    def test_scan_computes_correct_size_for_one_container(self, tmpdir):
        c1 = containers.Container(os.path.join(tmpdir.strpath, 'c1.h5'))
        c1.open()
        c1.set('utt-1', np.random.random((6, 6)).astype(np.float32))
        c1.set('utt-2', np.random.random((2, 6)).astype(np.float32))
        c1.set('utt-3', np.random.random((9, 6)).astype(np.float32))

        loader = partitioning.PartitioningContainerLoader(['utt-1', 'utt-2', 'utt-3'],
                                                          c1, '250', shuffle=True, seed=88)

        sizes = loader._scan()

        assert sizes == {
            'utt-1': 6 * 6 * np.dtype(np.float32).itemsize,
            'utt-2': 2 * 6 * np.dtype(np.float32).itemsize,
            'utt-3': 9 * 6 * np.dtype(np.float32).itemsize
        }

    def test_scan_computes_correct_size_for_multiple_containers(self, tmpdir):
        c1 = containers.Container(os.path.join(tmpdir.strpath, 'c1.h5'))
        c2 = containers.Container(os.path.join(tmpdir.strpath, 'c2.h5'))
        c3 = containers.Container(os.path.join(tmpdir.strpath, 'c3.h5'))
        c1.open()
        c1.set('utt-1', np.random.random((6, 6)).astype(np.float32))
        c1.set('utt-2', np.random.random((2, 6)).astype(np.float32))
        c1.set('utt-3', np.random.random((9, 6)).astype(np.float32))
        c2.open()
        c2.set('utt-1', np.random.random((2, 6)).astype(np.float32))
        c2.set('utt-2', np.random.random((1, 6)).astype(np.float32))
        c2.set('utt-3', np.random.random((4, 6)).astype(np.float32))
        c3.open()
        c3.set('utt-1', np.random.random((1, 6)).astype(np.float32))
        c3.set('utt-2', np.random.random((3, 6)).astype(np.float32))
        c3.set('utt-3', np.random.random((8, 6)).astype(np.float32))

        loader = partitioning.PartitioningContainerLoader(['utt-1', 'utt-2', 'utt-3'],
                                                          [c1, c2, c3], '1000', shuffle=True, seed=88)

        sizes = loader._scan()

        assert sizes == {
            'utt-1': (6 + 2 + 1) * 6 * np.dtype(np.float32).itemsize,
            'utt-2': (2 + 1 + 3) * 6 * np.dtype(np.float32).itemsize,
            'utt-3': (9 + 4 + 8) * 6 * np.dtype(np.float32).itemsize
        }

    def test_get_lengths_returns_correct_lengths_for_multiple_containers(self, tmpdir):
        c1 = containers.Container(os.path.join(tmpdir.strpath, 'c1.h5'))
        c2 = containers.Container(os.path.join(tmpdir.strpath, 'c2.h5'))
        c3 = containers.Container(os.path.join(tmpdir.strpath, 'c3.h5'))
        c1.open()
        c1.set('utt-1', np.random.random((6, 6)).astype(np.float32))
        c1.set('utt-2', np.random.random((2, 6)).astype(np.float32))
        c1.set('utt-3', np.random.random((9, 6)).astype(np.float32))
        c2.open()
        c2.set('utt-1', np.random.random((2, 6)).astype(np.float32))
        c2.set('utt-2', np.random.random((1, 6)).astype(np.float32))
        c2.set('utt-3', np.random.random((4, 6)).astype(np.float32))
        c3.open()
        c3.set('utt-1', np.random.random((1, 6)).astype(np.float32))
        c3.set('utt-2', np.random.random((3, 6)).astype(np.float32))
        c3.set('utt-3', np.random.random((8, 6)).astype(np.float32))

        loader = partitioning.PartitioningContainerLoader(['utt-1', 'utt-2', 'utt-3'],
                                                          [c1, c2, c3], '1000', shuffle=True, seed=88)

        lengths = loader._get_all_lengths()

        assert len(lengths) == 3
        assert lengths['utt-1'] == (6, 2, 1)
        assert lengths['utt-2'] == (2, 1, 3)
        assert lengths['utt-3'] == (9, 4, 8)

    def test_raises_error_if_utt_is_missing_in_container(self, tmpdir):
        c1 = containers.Container(os.path.join(tmpdir.strpath, 'c1.h5'))
        c1.open()
        c1.set('utt-1', np.random.random((6, 6)).astype(np.float32))
        c1.set('utt-3', np.random.random((9, 6)).astype(np.float32))

        with pytest.raises(ValueError):
            partitioning.PartitioningContainerLoader(['utt-1', 'utt-2', 'utt-3'],
                                                     c1, '250', shuffle=True, seed=88)

    def test_raises_error_if_utt_is_to_large_for_partition_size(self, tmpdir):
        c1 = containers.Container(os.path.join(tmpdir.strpath, 'c1.h5'))
        c1.open()
        c1.set('utt-1', np.random.random((6, 6)).astype(np.float32))
        # Needs 264
        c1.set('utt-3', np.random.random((11, 6)).astype(np.float32))

        with pytest.raises(ValueError):
            partitioning.PartitioningContainerLoader(['utt-1', 'utt-3'],
                                                     c1, '250', shuffle=True, seed=88)

    def test_reload_creates_correct_partitions(self, tmpdir):
        c1 = containers.Container(os.path.join(tmpdir.strpath, 'c1.h5'))
        c1.open()
        c1.set('utt-1', np.random.random((6, 6)).astype(np.float32))
        c1.set('utt-2', np.random.random((2, 6)).astype(np.float32))
        c1.set('utt-3', np.random.random((9, 6)).astype(np.float32))
        c1.set('utt-4', np.random.random((2, 6)).astype(np.float32))
        c1.set('utt-5', np.random.random((5, 6)).astype(np.float32))

        loader = partitioning.PartitioningContainerLoader(['utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'],
                                                          c1, '250', shuffle=False)

        assert len(loader.partitions) == 3
        assert loader.partitions[0].utt_ids == ['utt-1', 'utt-2']
        assert loader.partitions[0].utt_lengths == [(6,), (2,)]
        assert loader.partitions[0].size == 192
        assert loader.partitions[1].utt_ids == ['utt-3']
        assert loader.partitions[1].utt_lengths == [(9,)]
        assert loader.partitions[1].size == 216
        assert loader.partitions[2].utt_ids == ['utt-4', 'utt-5']
        assert loader.partitions[2].utt_lengths == [(2,), (5,)]
        assert loader.partitions[2].size == 168

    def test_reload_creates_no_partition_with_no_utterances(self, tmpdir):
        c1 = containers.Container(os.path.join(tmpdir.strpath, 'c1.h5'))
        c1.open()

        loader = partitioning.PartitioningContainerLoader([], c1, '250', shuffle=False)

        assert len(loader.partitions) == 0

    def test_reload_creates_different_partitions_on_second_run(self, tmpdir):
        c1 = containers.Container(os.path.join(tmpdir.strpath, 'c1.h5'))
        c1.open()
        c1.set('utt-1', np.random.random((6, 6)).astype(np.float32))
        c1.set('utt-2', np.random.random((2, 6)).astype(np.float32))
        c1.set('utt-3', np.random.random((9, 6)).astype(np.float32))
        c1.set('utt-4', np.random.random((2, 6)).astype(np.float32))
        c1.set('utt-5', np.random.random((5, 6)).astype(np.float32))

        loader = partitioning.PartitioningContainerLoader(['utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'],
                                                          c1, '250', shuffle=True, seed=100)

        partitions_one = loader.partitions
        loader.reload()
        partitions_two = loader.partitions

        len_changed = len(partitions_one) == len(partitions_two)

        if len_changed:
            assert True
        else:
            utt_ids_changed = False

            for x, y in zip(partitions_one, partitions_two):
                if x.utt_ids != y.utt_ids:
                    utt_ids_changed = True

            assert utt_ids_changed

    def test_load_partition_data(self, tmpdir):
        c1 = containers.Container(os.path.join(tmpdir.strpath, 'c1.h5'))
        c1.open()
        utt_1_data = np.random.random((6, 6)).astype(np.float32)
        utt_2_data = np.random.random((2, 6)).astype(np.float32)
        utt_3_data = np.random.random((9, 6)).astype(np.float32)
        utt_4_data = np.random.random((2, 6)).astype(np.float32)
        utt_5_data = np.random.random((5, 6)).astype(np.float32)
        c1.set('utt-1', utt_1_data)
        c1.set('utt-2', utt_2_data)
        c1.set('utt-3', utt_3_data)
        c1.set('utt-4', utt_4_data)
        c1.set('utt-5', utt_5_data)

        loader = partitioning.PartitioningContainerLoader(['utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'],
                                                          c1, '250', shuffle=False)

        part_1 = loader.load_partition_data(0)
        assert part_1.info.utt_ids == ['utt-1', 'utt-2']
        assert np.allclose(part_1.utt_data[0], utt_1_data)
        assert np.allclose(part_1.utt_data[1], utt_2_data)

        part_2 = loader.load_partition_data(1)
        assert part_2.info.utt_ids == ['utt-3']
        assert np.allclose(part_2.utt_data[0], utt_3_data)

        part_3 = loader.load_partition_data(2)
        assert part_3.info.utt_ids == ['utt-4', 'utt-5']
        assert np.allclose(part_3.utt_data[0], utt_4_data)
        assert np.allclose(part_3.utt_data[1], utt_5_data)


class TestPartitionInfo:

    def test_total_length_for_single_container(self):
        info = partitioning.PartitionInfo()
        info.utt_lengths = [(1,), (9,), (13,)]

        assert info.total_lengths() == (23,)

    def test_total_length_for_multiple_containers(self):
        info = partitioning.PartitionInfo()
        info.utt_lengths = [(1, 4), (9, 5), (13, 8)]

        assert info.total_lengths() == (23, 17)


class TestPartitioningFeatureIterator(object):

    def test_next_emits_no_features_if_file_is_empty(self, tmpdir):
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')

        features = tuple(PartitioningFeatureIterator(file, 120))
        assert 0 == len(features)

    def test_next_emits_no_features_if_data_set_is_empty(self, tmpdir):
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')
        file.create_dataset('utt-1', data=np.array([]))

        features = tuple(PartitioningFeatureIterator(file, 120))
        assert 0 == len(features)

    def test_next_emits_all_features_in_sequential_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')
        file.create_dataset('utt-1', data=ds1)
        file.create_dataset('utt-2', data=ds2)

        features = tuple(PartitioningFeatureIterator(file, 120, shuffle=False))

        assert 5 == len(features)

        self.assert_features_equal(('utt-1', 0, [0.1, 0.1, 0.1, 0.1, 0.1]), features[0])
        self.assert_features_equal(('utt-1', 1, [0.2, 0.2, 0.2, 0.2, 0.2]), features[1])
        self.assert_features_equal(('utt-2', 0, [0.3, 0.3, 0.3, 0.3, 0.3]), features[2])
        self.assert_features_equal(('utt-2', 1, [0.4, 0.4, 0.4, 0.4, 0.4]), features[3])
        self.assert_features_equal(('utt-2', 2, [0.5, 0.5, 0.5, 0.5, 0.5]), features[4])

    def test_next_emits_all_features_in_random_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')
        file.create_dataset('utt-1', data=ds1)
        file.create_dataset('utt-2', data=ds2)
        file.create_dataset('utt-3', data=ds3)

        features = tuple(PartitioningFeatureIterator(file, 120, shuffle=True, seed=16))

        assert 6 == len(features)

        self.assert_features_equal(('utt-1', 1, [0.2, 0.2, 0.2, 0.2, 0.2]), features[0])
        self.assert_features_equal(('utt-3', 0, [0.6, 0.6, 0.6, 0.6, 0.6]), features[1])
        self.assert_features_equal(('utt-1', 0, [0.1, 0.1, 0.1, 0.1, 0.1]), features[2])
        self.assert_features_equal(('utt-2', 2, [0.5, 0.5, 0.5, 0.5, 0.5]), features[3])
        self.assert_features_equal(('utt-2', 0, [0.3, 0.3, 0.3, 0.3, 0.3]), features[4])
        self.assert_features_equal(('utt-2', 1, [0.4, 0.4, 0.4, 0.4, 0.4]), features[5])

    def test_next_emits_features_only_from_included_ds_in_sequential_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')
        file.create_dataset('utt-1', data=ds1)
        file.create_dataset('utt-2', data=ds2)
        file.create_dataset('utt-3', data=ds3)

        features = tuple(PartitioningFeatureIterator(file, 120, shuffle=False, includes=['utt-1', 'utt-3', 'unknown']))

        assert 4 == len(features)

        self.assert_features_equal(('utt-1', 0, [0.1, 0.1, 0.1, 0.1, 0.1]), features[0])
        self.assert_features_equal(('utt-1', 1, [0.2, 0.2, 0.2, 0.2, 0.2]), features[1])
        self.assert_features_equal(('utt-3', 0, [0.6, 0.6, 0.6, 0.6, 0.6]), features[2])
        self.assert_features_equal(('utt-3', 1, [0.7, 0.7, 0.7, 0.7, 0.7]), features[3])

    def test_next_emits_features_only_from_included_ds_in_random_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')
        file.create_dataset('utt-1', data=ds1)
        file.create_dataset('utt-2', data=ds2)
        file.create_dataset('utt-3', data=ds3)

        features = tuple(PartitioningFeatureIterator(file, 120, shuffle=True, seed=16,
                                                     includes=['utt-1', 'utt-3', 'unknown']))

        assert 4 == len(features)

        self.assert_features_equal(('utt-3', 0, [0.6, 0.6, 0.6, 0.6, 0.6]), features[0])
        self.assert_features_equal(('utt-1', 0, [0.1, 0.1, 0.1, 0.1, 0.1]), features[1])
        self.assert_features_equal(('utt-1', 1, [0.2, 0.2, 0.2, 0.2, 0.2]), features[2])
        self.assert_features_equal(('utt-3', 1, [0.7, 0.7, 0.7, 0.7, 0.7]), features[3])

    def test_next_emits_features_without_excluded_in_sequential_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')
        file.create_dataset('utt-1', data=ds1)
        file.create_dataset('utt-2', data=ds2)
        file.create_dataset('utt-3', data=ds3)

        features = tuple(PartitioningFeatureIterator(file, 120, shuffle=False, excludes=['utt-2', 'unknown']))

        assert 4 == len(features)

        self.assert_features_equal(('utt-1', 0, [0.1, 0.1, 0.1, 0.1, 0.1]), features[0])
        self.assert_features_equal(('utt-1', 1, [0.2, 0.2, 0.2, 0.2, 0.2]), features[1])
        self.assert_features_equal(('utt-3', 0, [0.6, 0.6, 0.6, 0.6, 0.6]), features[2])
        self.assert_features_equal(('utt-3', 1, [0.7, 0.7, 0.7, 0.7, 0.7]), features[3])

    def test_next_emits_features_without_excluded_in_random_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')
        file.create_dataset('utt-1', data=ds1)
        file.create_dataset('utt-2', data=ds2)
        file.create_dataset('utt-3', data=ds3)

        features = tuple(PartitioningFeatureIterator(file, 120, shuffle=True, seed=16, excludes=['utt-2', 'unknown']))

        assert 4 == len(features)

        self.assert_features_equal(('utt-3', 0, [0.6, 0.6, 0.6, 0.6, 0.6]), features[0])
        self.assert_features_equal(('utt-1', 0, [0.1, 0.1, 0.1, 0.1, 0.1]), features[1])
        self.assert_features_equal(('utt-1', 1, [0.2, 0.2, 0.2, 0.2, 0.2]), features[2])
        self.assert_features_equal(('utt-3', 1, [0.7, 0.7, 0.7, 0.7, 0.7]), features[3])

    def test_next_emits_features_only_from_included_ds_ignoring_filter_in_sequential_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')
        file.create_dataset('utt-1', data=ds1)
        file.create_dataset('utt-2', data=ds2)
        file.create_dataset('utt-3', data=ds3)

        features = tuple(PartitioningFeatureIterator(file, 120, shuffle=False, includes=['utt-1', 'utt-3', 'unknown'],
                                                     excludes=['utt-1']))

        assert 4 == len(features)

        self.assert_features_equal(('utt-1', 0, [0.1, 0.1, 0.1, 0.1, 0.1]), features[0])
        self.assert_features_equal(('utt-1', 1, [0.2, 0.2, 0.2, 0.2, 0.2]), features[1])
        self.assert_features_equal(('utt-3', 0, [0.6, 0.6, 0.6, 0.6, 0.6]), features[2])
        self.assert_features_equal(('utt-3', 1, [0.7, 0.7, 0.7, 0.7, 0.7]), features[3])

    def test_next_emits_features_only_from_included_ds_ignoring_filter_in_random_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')
        file.create_dataset('utt-1', data=ds1)
        file.create_dataset('utt-2', data=ds2)
        file.create_dataset('utt-3', data=ds3)

        features = tuple(PartitioningFeatureIterator(file, 120, shuffle=True, seed=16,
                                                     includes=['utt-1', 'utt-3', 'unknown'], excludes=['utt-1']))

        assert 4 == len(features)

        self.assert_features_equal(('utt-3', 0, [0.6, 0.6, 0.6, 0.6, 0.6]), features[0])
        self.assert_features_equal(('utt-1', 0, [0.1, 0.1, 0.1, 0.1, 0.1]), features[1])
        self.assert_features_equal(('utt-1', 1, [0.2, 0.2, 0.2, 0.2, 0.2]), features[2])
        self.assert_features_equal(('utt-3', 1, [0.7, 0.7, 0.7, 0.7, 0.7]), features[3])

    def test_next_emits_all_features_if_partition_spans_multiple_data_sets_in_sequential_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')
        file.create_dataset('utt-1', data=ds1)
        file.create_dataset('utt-2', data=ds2)
        file.create_dataset('utt-3', data=ds3)

        iterator = PartitioningFeatureIterator(file, 240, shuffle=False)
        features = tuple(iterator)

        assert 7 == len(features)

        self.assert_features_equal(('utt-1', 0, [0.1, 0.1, 0.1, 0.1, 0.1]), features[0])
        self.assert_features_equal(('utt-1', 1, [0.2, 0.2, 0.2, 0.2, 0.2]), features[1])
        self.assert_features_equal(('utt-2', 0, [0.3, 0.3, 0.3, 0.3, 0.3]), features[2])
        self.assert_features_equal(('utt-2', 1, [0.4, 0.4, 0.4, 0.4, 0.4]), features[3])
        self.assert_features_equal(('utt-2', 2, [0.5, 0.5, 0.5, 0.5, 0.5]), features[4])
        self.assert_features_equal(('utt-3', 0, [0.6, 0.6, 0.6, 0.6, 0.6]), features[5])
        self.assert_features_equal(('utt-3', 1, [0.7, 0.7, 0.7, 0.7, 0.7]), features[6])

    def test_next_emits_all_features_if_partition_spans_multiple_data_sets_in_random_order(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')
        file.create_dataset('utt-1', data=ds1)
        file.create_dataset('utt-2', data=ds2)
        file.create_dataset('utt-3', data=ds3)

        iterator = PartitioningFeatureIterator(file, 240, shuffle=True, seed=42)
        features = tuple(iterator)

        assert 7 == len(features)

        self.assert_features_equal(('utt-3', 1, [0.7, 0.7, 0.7, 0.7, 0.7]), features[0])
        self.assert_features_equal(('utt-1', 0, [0.1, 0.1, 0.1, 0.1, 0.1]), features[1])
        self.assert_features_equal(('utt-1', 1, [0.2, 0.2, 0.2, 0.2, 0.2]), features[2])
        self.assert_features_equal(('utt-3', 0, [0.6, 0.6, 0.6, 0.6, 0.6]), features[3])
        self.assert_features_equal(('utt-2', 0, [0.3, 0.3, 0.3, 0.3, 0.3]), features[4])
        self.assert_features_equal(('utt-2', 2, [0.5, 0.5, 0.5, 0.5, 0.5]), features[5])
        self.assert_features_equal(('utt-2', 1, [0.4, 0.4, 0.4, 0.4, 0.4]), features[6])

    def test_partitioning_empty_file_emits_zero_partitions(self, tmpdir):
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')

        iterator = PartitioningFeatureIterator(file, 100, shuffle=True, seed=42)

        assert 0 == len(iterator._partitions)

    def test_partitioning_empty_ds_emits_zero_partitions(self, tmpdir):
        ds1 = np.array([])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')
        file.create_dataset('utt-1', data=ds1)

        iterator = PartitioningFeatureIterator(file, 100, shuffle=True, seed=42)

        assert 0 == len(iterator._partitions)

    def test_partitioning_all_data_fits_exactly_in_one(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        ds3 = np.array([[0.6, 0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')
        file.create_dataset('utt-1', data=ds1)
        file.create_dataset('utt-2', data=ds2)
        file.create_dataset('utt-3', data=ds3)

        iterator = PartitioningFeatureIterator(file, 320, shuffle=True, seed=42)

        assert 1 == len(iterator._partitions)
        assert (('utt-1', 0), ('utt-3', 2)) in iterator._partitions

    def test_partitioning_partition_size_not_divisible_by_record_size_without_remainder(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')
        file.create_dataset('utt-1', data=ds1)
        file.create_dataset('utt-2', data=ds2)

        iterator = PartitioningFeatureIterator(file, 81, shuffle=True, seed=42)  # one byte more than ds1

        assert 3 == len(iterator._partitions)
        assert (('utt-1', 1), ('utt-1', 2)) in iterator._partitions
        assert (('utt-2', 0), ('utt-2', 2)) in iterator._partitions
        assert (('utt-2', 2), ('utt-1', 1)) in iterator._partitions

    def test_partitioning_remaining_space_filled_with_next_data_set(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')
        file.create_dataset('utt-1', data=ds1)
        file.create_dataset('utt-2', data=ds2)

        iterator = PartitioningFeatureIterator(file, 120, shuffle=True, seed=32)

        assert 2 == len(iterator._partitions)
        assert (('utt-1', 0), ('utt-2', 1)) in iterator._partitions
        assert (('utt-2', 1), ('utt-2', 3)) in iterator._partitions

    def test_partitioning_copes_with_varying_record_sizes(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        ds2 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')
        file.create_dataset('utt-1', data=ds1)
        file.create_dataset('utt-2', data=ds2)

        iterator = PartitioningFeatureIterator(file, 120, shuffle=True, seed=12)

        assert 3 == len(iterator._partitions)
        assert (('utt-1', 0), ('utt-1', 2)) in iterator._partitions
        assert (('utt-2', 0), ('utt-2', 2)) in iterator._partitions
        assert (('utt-2', 2), ('utt-2', 3)) in iterator._partitions

    def test_partitioning_raises_error_if_record_bigger_than_partition_size(self, tmpdir):
        ds1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1]])
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')
        file.create_dataset('utt-1', data=ds1)

        with pytest.raises(ValueError):
            PartitioningFeatureIterator(file, 1)

    @staticmethod
    def assert_features_equal(expected, actual):
        if expected[0] != actual[0] or expected[1] != actual[1] or not np.allclose(expected[2], actual[2]):
            raise AssertionError('Expected {0} but got {1} instead'.format(expected, actual))
