import os

import numpy as np

from audiomate.corpus import assets
from audiomate import feeding

import pytest
from tests import resources


class TestDataset:

    def test_init_throws_error_when_no_container_is_given(self):
        corpus = resources.create_dataset()

        with pytest.raises(ValueError):
            feeding.Dataset(corpus, [])

    def test_init_with_corpus(self):
        corpus = resources.create_dataset()
        it = feeding.Dataset(corpus, [assets.Container('blub')])
        assert set(it.utt_ids) == set(corpus.utterances.keys())

    def test_init_with_utterance_list(self):
        it = feeding.Dataset(['utt-1', 'utt-2'], [assets.Container('blub')])
        assert set(it.utt_ids) == {'utt-1', 'utt-2'}


@pytest.fixture
def sample_frame_dataset(tmpdir):
    inputs_path = os.path.join(tmpdir.strpath, 'inputs.hdf5')
    targets_path = os.path.join(tmpdir.strpath, 'targets.hdf5')

    corpus = resources.create_dataset()
    container_inputs = assets.Container(inputs_path)
    container_targets = assets.Container(targets_path)

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

    def test_get_utt_regions(self, sample_frame_dataset):
        regions = sample_frame_dataset.get_utt_regions()

        assert regions[0][0] == 0
        assert regions[0][1] == 5
        assert np.array_equal(regions[0][2][0][()], np.arange(20).reshape(5, 4))
        assert np.array_equal(regions[0][2][1][()], np.arange(20).reshape(5, 4) + 10)

        assert regions[2][0] == 12
        assert regions[2][1] == 9
        assert np.array_equal(regions[2][2][0][()], np.arange(36).reshape(9, 4))
        assert np.array_equal(regions[2][2][1][()], np.arange(36).reshape(9, 4) + 10)

        assert regions[4][0] == 23
        assert regions[4][1] == 4
        assert np.array_equal(regions[4][2][0][()], np.arange(16).reshape(4, 4))
        assert np.array_equal(regions[4][2][1][()], np.arange(16).reshape(4, 4) + 10)

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
