import os

import numpy as np

from audiomate import containers
from audiomate.corpus import subset
from audiomate import feeding
from audiomate.processing import pipeline

import pytest
from tests import resources


@pytest.fixture
def container_dim_x_4(tmpdir):
    inputs_path = os.path.join(tmpdir.strpath, 'inputs.hdf5')

    cnt = containers.FeatureContainer(inputs_path)
    cnt.open()

    cnt.frame_size = 4
    cnt.hop_size = 2
    cnt.sampling_rate = 16000

    cnt.set('utt-1', np.arange(60).reshape(15, 4))
    cnt.set('utt-2', np.arange(80).reshape(20, 4))
    cnt.set('utt-3', np.arange(44).reshape(11, 4))
    cnt.set('utt-4', np.arange(12).reshape(3, 4))
    cnt.set('utt-5', np.arange(16).reshape(4, 4))

    return cnt


@pytest.fixture
def container_dim_x(tmpdir):
    inputs_path = os.path.join(tmpdir.strpath, 'outputs.hdf5')

    cnt = containers.Container(inputs_path)
    cnt.open()

    cnt.set('utt-1', np.arange(6))
    cnt.set('utt-2', np.arange(8))
    cnt.set('utt-3', np.arange(4))
    cnt.set('utt-4', np.arange(2))
    cnt.set('utt-5', np.arange(6))

    return cnt


class TestDataset:

    def test_init_throws_error_when_no_container_is_given(self):
        corpus = resources.create_dataset()

        with pytest.raises(ValueError):
            feeding.Dataset(corpus, [])

    def test_init_with_corpus(self, tmpdir):
        c = containers.Container(os.path.join(tmpdir.strpath, 'test.h5'))
        c.open()
        c.set('utt-1', data=np.arange(20))
        c.set('utt-2', data=np.arange(20))
        c.set('utt-3', data=np.arange(20))
        c.set('utt-4', data=np.arange(20))
        c.set('utt-5', data=np.arange(20))

        corpus = resources.create_dataset()
        it = feeding.Dataset(corpus, [c])
        assert it.utt_ids == ['utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5']

    def test_init_with_corpus_view(self, tmpdir):
        c = containers.Container(os.path.join(tmpdir.strpath, 'test.h5'))
        c.open()
        c.set('utt-1', data=np.arange(20))
        c.set('utt-2', data=np.arange(20))
        c.set('utt-3', data=np.arange(20))
        c.set('utt-4', data=np.arange(20))
        c.set('utt-5', data=np.arange(20))

        corpus = resources.create_dataset()
        subview = subset.Subview(corpus, filter_criteria=[
            subset.MatchingUtteranceIdxFilter(utterance_idxs={'utt-1', 'utt-2', 'utt-4'})
        ])

        it = feeding.Dataset(subview, [c])
        assert it.utt_ids == ['utt-1', 'utt-2', 'utt-4']

    def test_init_with_utterance_list(self, tmpdir):
        c = containers.Container(os.path.join(tmpdir.strpath, 'test.h5'))
        c.open()
        c.set('utt-1', data=np.arange(20))
        c.set('utt-2', data=np.arange(20))

        it = feeding.Dataset(['utt-1', 'utt-2'], [c])
        assert it.utt_ids == ['utt-1', 'utt-2']

    def test_init_missing_utterance_in_container_raises_error(self, tmpdir):
        c = containers.Container(os.path.join(tmpdir.strpath, 'test.h5'))
        c.open()
        c.set('utt-1', data=np.arange(20))
        c.set('utt-3', data=np.arange(20))

        with pytest.raises(ValueError):
            feeding.Dataset(['utt-1', 'utt-2', 'utt-3'], [containers.Container('blub')])

    def test_container_has_utterances(self, tmpdir):
        c = containers.Container(os.path.join(tmpdir.strpath, 'test.h5'))
        c.open()
        c.set('utt-1', data=np.arange(20))
        c.set('utt-2', data=np.arange(20))
        c.set('utt-3', data=np.arange(20))

        assert feeding.Dataset.container_has_utterances(c, ['utt-1', 'utt-2', 'utt-3'])

    def test_container_has_utterances_returns_false_if_one_is_missing(self, tmpdir):
        c = containers.Container(os.path.join(tmpdir.strpath, 'test.h5'))
        c.open()
        c.set('utt-1', data=np.arange(20))
        c.set('utt-3', data=np.arange(20))

        assert not feeding.Dataset.container_has_utterances(c, ['utt-1', 'utt-2', 'utt-3'])


class AddTransform(pipeline.Computation):
    def __init__(self, value, parent=None, name=None):
        super(AddTransform, self).__init__(parent=parent, name=name)
        self.value = value

    def compute(self, chunk, sampling_rate, corpus=None, utterance=None):
        return chunk.data + self.value


class TestUtteranceDataset:

    def test_init_with_corpus(self, container_dim_x_4, container_dim_x):
        corpus = resources.create_dataset()
        ds = feeding.UtteranceDataset(corpus, [container_dim_x_4, container_dim_x])

        assert ds.utt_ids == ['utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5']

    def test_init_with_list_of_utt_ids(self, container_dim_x_4, container_dim_x):
        ds = feeding.UtteranceDataset(['utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'],
                                      [container_dim_x_4, container_dim_x])

        assert ds.utt_ids == ['utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5']

    def test_len(self, container_dim_x_4, container_dim_x):
        ds = feeding.UtteranceDataset(['utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'],
                                      [container_dim_x_4, container_dim_x])

        assert len(ds) == 5

    def test_get_item(self, container_dim_x_4, container_dim_x):
        ds = feeding.UtteranceDataset(['utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'],
                                      [container_dim_x_4, container_dim_x])

        sample = ds[2]

        assert len(sample) == 4
        assert np.array_equal(sample[0], np.arange(44).reshape(11, 4))
        assert sample[1] == 11
        assert np.array_equal(sample[2], np.arange(4))
        assert sample[3] == 4

    def test_get_item_padded(self, container_dim_x_4, container_dim_x):
        ds = feeding.UtteranceDataset(['utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'],
                                      [container_dim_x_4, container_dim_x], pad=True)

        sample = ds[2]

        assert len(sample) == 4
        assert np.array_equal(sample[0], np.pad(np.arange(44).reshape(11, 4), ((0, 9), (0, 0)), mode='constant'))
        assert sample[1] == 11
        assert np.array_equal(sample[2], np.pad(np.arange(4), (0, 4), mode='constant'))
        assert sample[3] == 4

    def test_get_item_padded_when_no_padding_required(self, container_dim_x_4, container_dim_x):
        ds = feeding.UtteranceDataset(['utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'],
                                      [container_dim_x_4, container_dim_x], pad=True)

        sample = ds[1]

        assert len(sample) == 4
        assert np.array_equal(sample[0], np.arange(80).reshape(20, 4))
        assert sample[1] == 20
        assert np.array_equal(sample[2], np.arange(8))
        assert sample[3] == 8

    def test_longest_utterances_per_container(self, container_dim_x_4, container_dim_x):
        ds = feeding.UtteranceDataset(['utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'],
                                      [container_dim_x_4, container_dim_x])

        lengths = ds.longest_utterances_per_container()

        assert len(lengths) == 2
        assert lengths[0] == 20
        assert lengths[1] == 8

    def test_get_item_with_transfrom(self, container_dim_x_4, container_dim_x):
        transform = AddTransform(value=3.0)

        ds = feeding.UtteranceDataset(
            ['utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'],
            [container_dim_x_4, container_dim_x],
            transform=[transform, None]
        )

        sample = ds[2]

        assert len(sample) == 4
        assert np.array_equal(sample[0], np.arange(44).reshape(11, 4) + 3.0)
        assert sample[1] == 11
        assert np.array_equal(sample[2], np.arange(4))
        assert sample[3] == 4

    def test_init_with_transform_for_non_feature_container_raises_error(
            self, container_dim_x_4, container_dim_x):
        transform = AddTransform(value=3.0)

        with pytest.raises(ValueError):
            feeding.UtteranceDataset(
                ['utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'],
                [container_dim_x_4, container_dim_x],
                transform=[transform, transform]
            )

    def test_init_with_missing_transforms_raises_error(self, container_dim_x_4, container_dim_x):
        transform = AddTransform(value=3.0)

        with pytest.raises(ValueError):
            feeding.UtteranceDataset(
                ['utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'],
                [container_dim_x_4, container_dim_x],
                transform=[transform]
            )


@pytest.fixture
def sample_multi_frame_dataset(tmpdir):
    inputs_path = os.path.join(tmpdir.strpath, 'inputs.hdf5')
    targets_path = os.path.join(tmpdir.strpath, 'targets.hdf5')

    corpus = resources.create_dataset()
    container_inputs = containers.Container(inputs_path)
    container_targets = containers.Container(targets_path)

    container_inputs.open()
    container_targets.open()

    container_inputs.set('utt-1', np.arange(60).reshape(15, 4))
    container_inputs.set('utt-2', np.arange(80).reshape(20, 4))
    container_inputs.set('utt-3', np.arange(44).reshape(11, 4))
    container_inputs.set('utt-4', np.arange(12).reshape(3, 4))
    container_inputs.set('utt-5', np.arange(16).reshape(4, 4))

    container_targets.set('utt-1', np.arange(30).reshape(15, 2))
    container_targets.set('utt-2', np.arange(40).reshape(20, 2))
    container_targets.set('utt-3', np.arange(22).reshape(11, 2))
    container_targets.set('utt-4', np.arange(6).reshape(3, 2))
    container_targets.set('utt-5', np.arange(8).reshape(4, 2))

    return feeding.MultiFrameDataset(corpus, [container_inputs, container_targets], 4)


class TestMultiFrameDataset:

    def test_raises_error_if_frames_per_chunk_is_smaller_than_one(self):
        with pytest.raises(ValueError):
            feeding.MultiFrameDataset(['utt-1'], [], 0)

    def test_partitioned_iterator(self, sample_multi_frame_dataset):
        it = sample_multi_frame_dataset.partitioned_iterator('960', shuffle=True, seed=12)

        assert isinstance(it, feeding.MultiFrameIterator)
        assert it.return_length == sample_multi_frame_dataset.return_length
        assert it.pad == sample_multi_frame_dataset.pad
        assert it.frames_per_chunk == sample_multi_frame_dataset.frames_per_chunk
        assert it.containers == sample_multi_frame_dataset.containers
        assert it.utt_ids == sample_multi_frame_dataset.utt_ids
        assert it.shuffle
        assert it.partition_size == '960'

    def test_get_utt_regions(self, sample_multi_frame_dataset):
        regions = sample_multi_frame_dataset.get_utt_regions()

        assert len(regions) == 5

        assert regions[0][0] == 0
        assert regions[0][1] == 4

        assert regions[1][0] == 4
        assert regions[1][1] == 5

        assert regions[2][0] == 9
        assert regions[2][1] == 3

        assert regions[3][0] == 12
        assert regions[3][1] == 1

        assert regions[4][0] == 13
        assert regions[4][1] == 1

    def test_get_length(self, sample_multi_frame_dataset):
        assert len(sample_multi_frame_dataset) == 14

    def test_get_item_in_the_middle_of_an_utterance(self, sample_multi_frame_dataset):
        assert len(sample_multi_frame_dataset[2]) == 2
        assert np.array_equal(sample_multi_frame_dataset[2][0], np.arange(16).reshape(4, 4) + 32)
        assert np.array_equal(sample_multi_frame_dataset[2][1], np.arange(8).reshape(4, 2) + 16)

    def test_get_item_at_the_begin_of_an_utterance(self, sample_multi_frame_dataset):
        assert len(sample_multi_frame_dataset[4]) == 2
        assert np.array_equal(sample_multi_frame_dataset[4][0], np.arange(16).reshape(4, 4))
        assert np.array_equal(sample_multi_frame_dataset[4][1], np.arange(8).reshape(4, 2))

    def test_get_item_at_the_end_of_an_utterance(self, sample_multi_frame_dataset):
        assert len(sample_multi_frame_dataset[11]) == 2
        assert np.array_equal(sample_multi_frame_dataset[11][0], np.arange(12).reshape(3, 4) + 32)
        assert np.array_equal(sample_multi_frame_dataset[11][1], np.arange(6).reshape(3, 2) + 16)

    def test_return_correct_length_for_chunk_with_full_size(self, sample_multi_frame_dataset):
        ds_length_enabled = feeding.MultiFrameDataset(sample_multi_frame_dataset.utt_ids,
                                                      sample_multi_frame_dataset.containers,
                                                      4,
                                                      return_length=True)

        assert len(ds_length_enabled[9]) == 3
        assert ds_length_enabled[9][2] == 4

    def test_return_correct_length_for_chunk_at_end_of_utterance(self, sample_multi_frame_dataset):
        ds_length_enabled = feeding.MultiFrameDataset(sample_multi_frame_dataset.utt_ids,
                                                      sample_multi_frame_dataset.containers,
                                                      4,
                                                      return_length=True)

        assert len(ds_length_enabled[11]) == 3
        assert ds_length_enabled[11][2] == 3

    def test_pads_shorter_chunks_with_zeros(self, sample_multi_frame_dataset):
        ds_pad_enabled = feeding.MultiFrameDataset(sample_multi_frame_dataset.utt_ids,
                                                   sample_multi_frame_dataset.containers,
                                                   4,
                                                   pad=True)

        assert len(ds_pad_enabled[11]) == 3
        exp = np.pad(np.arange(12).reshape(3, 4) + 32, ((0, 1), (0, 0)), mode='constant', constant_values=0)
        assert np.array_equal(ds_pad_enabled[11][0], exp)
        exp = np.pad(np.arange(6).reshape(3, 2) + 16, ((0, 1), (0, 0)), mode='constant', constant_values=0)
        assert np.array_equal(ds_pad_enabled[11][1], exp)
        assert ds_pad_enabled[11][2] == 3


@pytest.fixture
def sample_frame_dataset(tmpdir):
    inputs_path = os.path.join(tmpdir.strpath, 'inputs.hdf5')
    targets_path = os.path.join(tmpdir.strpath, 'targets.hdf5')

    corpus = resources.create_dataset()
    container_inputs = containers.Container(inputs_path)
    container_targets = containers.Container(targets_path)

    container_inputs.open()
    container_targets.open()

    container_inputs.set('utt-1', np.arange(20).reshape(5, 4))
    container_inputs.set('utt-2', np.arange(28).reshape(7, 4))
    container_inputs.set('utt-3', np.arange(36).reshape(9, 4))
    container_inputs.set('utt-4', np.arange(8).reshape(2, 4))
    container_inputs.set('utt-5', np.arange(16).reshape(4, 4))

    container_targets.set('utt-1', np.arange(20).reshape(5, 4) + 10)
    container_targets.set('utt-2', np.arange(28).reshape(7, 4) + 10)
    container_targets.set('utt-3', np.arange(36).reshape(9, 4) + 10)
    container_targets.set('utt-4', np.arange(8).reshape(2, 4) + 10)
    container_targets.set('utt-5', np.arange(16).reshape(4, 4) + 10)

    return feeding.FrameDataset(corpus, [container_inputs, container_targets])


class TestFrameDataset:

    def test_partitioned_iterator(self, sample_frame_dataset):
        it = sample_frame_dataset.partitioned_iterator('960', shuffle=True, seed=12)

        assert isinstance(it, feeding.FrameIterator)
        assert it.containers == sample_frame_dataset.containers
        assert it.utt_ids == sample_frame_dataset.utt_ids
        assert it.shuffle
        assert it.partition_size == '960'

    def test_get_length(self, sample_frame_dataset):
        assert len(sample_frame_dataset) == 27

    def test_get_item(self, sample_frame_dataset):
        assert len(sample_frame_dataset[4]) == 2
        assert np.array_equal(sample_frame_dataset[4][0], np.array([16, 17, 18, 19]))
        assert np.array_equal(sample_frame_dataset[4][1], np.array([16, 17, 18, 19]) + 10)

        assert len(sample_frame_dataset[12]) == 2
        assert np.array_equal(sample_frame_dataset[12][0], np.array([0, 1, 2, 3]))
        assert np.array_equal(sample_frame_dataset[12][1], np.array([0, 1, 2, 3]) + 10)

        assert len(sample_frame_dataset[22]) == 2
        assert np.array_equal(sample_frame_dataset[22][0], np.array([4, 5, 6, 7]))
        assert np.array_equal(sample_frame_dataset[22][1], np.array([4, 5, 6, 7]) + 10)

        assert len(sample_frame_dataset[26]) == 2
        assert np.array_equal(sample_frame_dataset[26][0], np.array([12, 13, 14, 15]))
        assert np.array_equal(sample_frame_dataset[26][1], np.array([12, 13, 14, 15]) + 10)
