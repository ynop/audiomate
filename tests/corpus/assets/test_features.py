import os

import h5py
import numpy as np
import pytest

from audiomate.corpus import assets
from audiomate.corpus.assets.features import PartitioningFeatureIterator
from tests import resources


@pytest.fixture()
def sample_feature_container():
    container = assets.FeatureContainer(resources.get_resource_path(['sample_files', 'feat_container']))
    container.open()
    yield container
    container.close()


class TestFeatureContainer:

    def test_frame_size(self, sample_feature_container):
        assert sample_feature_container.frame_size == 400

    def test_hop_size(self, sample_feature_container):
        assert sample_feature_container.hop_size == 160

    def test_sampling_rate(self, sample_feature_container):
        assert sample_feature_container.sampling_rate == 16000

    def test_stats_per_utterance(self, sample_feature_container):
        utt_stats = sample_feature_container.stats_per_utterance()

        assert utt_stats['utt-1'].min == pytest.approx(0.0071605651933048797)
        assert utt_stats['utt-1'].max == pytest.approx(0.9967182746569494)
        assert utt_stats['utt-1'].mean == pytest.approx(0.51029100520776705)
        assert utt_stats['utt-1'].var == pytest.approx(0.079222738766221268)
        assert utt_stats['utt-1'].num == 100

        assert utt_stats['utt-2'].min == pytest.approx(0.01672865642756316)
        assert utt_stats['utt-2'].max == pytest.approx(0.99394433783429104)
        assert utt_stats['utt-2'].mean == pytest.approx(0.46471979908661543)
        assert utt_stats['utt-2'].var == pytest.approx(0.066697466410977804)
        assert utt_stats['utt-2'].num == 65

        assert utt_stats['utt-3'].min == pytest.approx(0.014999482706963607)
        assert utt_stats['utt-3'].max == pytest.approx(0.99834417857609881)
        assert utt_stats['utt-3'].mean == pytest.approx(0.51042690965262705)
        assert utt_stats['utt-3'].var == pytest.approx(0.071833200069641057)
        assert utt_stats['utt-3'].num == 220

    def test_stats_per_utterance_not_open(self, sample_feature_container):
        sample_feature_container.close()

        with pytest.raises(ValueError):
            sample_feature_container.stats_per_utterance()

    def test_stats(self, sample_feature_container):
        stats = sample_feature_container.stats()

        assert stats.min == pytest.approx(0.0071605651933048797)
        assert stats.max == pytest.approx(0.99834417857609881)
        assert stats.mean == pytest.approx(0.50267482489606408)
        assert stats.var == pytest.approx(0.07317811077366114)

    def test_stats_not_open(self, sample_feature_container):
        sample_feature_container.close()

        with pytest.raises(ValueError):
            sample_feature_container.stats()

    def test_append(self, tmpdir):
        container = assets.FeatureContainer(os.path.join(tmpdir.strpath, 'container'))
        container.open()

        data = np.arange(100).reshape(20, 5)

        container.append('utt-1', data[:8])
        container.append('utt-1', data[8:])

        res = container.get('utt-1', mem_map=False)

        assert np.array_equal(data, res)

        container.close()

    def test_append_with_different_dimension_raises_error(self, tmpdir):
        container = assets.FeatureContainer(os.path.join(tmpdir.strpath, 'container'))
        container.open()

        container.append('utt-1', np.arange(20).reshape(5, 2, 2))

        with pytest.raises(ValueError):
            container.append('utt-1', np.arange(42).reshape(7, 2, 3))

        container.close()


class TestPartitioningFeatureIterator(object):

    def test_partition_size_in_bytes_specified_as_int(self, tmpdir):
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')

        iterator = PartitioningFeatureIterator(file, 1024)

        assert 1024 == iterator._partition_size

    def test_partition_size_in_bytes(self, tmpdir):
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')

        iterator = PartitioningFeatureIterator(file, '1024')

        assert 1024 == iterator._partition_size

    def test_partition_size_in_kibibytes(self, tmpdir):
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')

        iterator = PartitioningFeatureIterator(file, '2k')

        assert 2 * 1024 == iterator._partition_size

    def test_partition_size_in_mebibytes(self, tmpdir):
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')

        iterator = PartitioningFeatureIterator(file, '2m')
        assert 2 * 1024 * 1024 == iterator._partition_size

    def test_partition_size_in_gibibytes(self, tmpdir):
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')

        iterator = PartitioningFeatureIterator(file, '2g')
        assert 2 * 1024 * 1024 * 1024 == iterator._partition_size

    def test_partition_size_in_gibibytes_with_capital_g(self, tmpdir):
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')

        iterator = PartitioningFeatureIterator(file, '2G')
        assert 2 * 1024 * 1024 * 1024 == iterator._partition_size

    def test_partition_size_fractions_of_bytes_are_ignored(self, tmpdir):
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')

        iterator = PartitioningFeatureIterator(file, '1.1')
        assert 1 == iterator._partition_size

    def test_partition_size_half_a_gibibyte(self, tmpdir):
        file_path = os.path.join(tmpdir.strpath, 'features.h5')
        file = h5py.File(file_path, 'w')

        iterator = PartitioningFeatureIterator(file, '0.5g')
        assert 512 * 1024 * 1024 == iterator._partition_size

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
