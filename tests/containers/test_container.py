import os

import numpy as np

from audiomate import containers
import pytest

from tests import resources


@pytest.fixture()
def sample_container():
    container_path = resources.get_resource_path(
        ['sample_files', 'feat_container']
    )
    sample_container = containers.Container(container_path)
    sample_container.open()
    yield sample_container
    sample_container.close()


class TestContainer:

    def test_is_open(self, sample_container):
        assert sample_container.is_open()

    def test_is_open_returns_false(self, sample_container):
        sample_container.close()
        assert not sample_container.is_open()

    def test_keys(self, sample_container):
        assert sample_container.keys() == ['utt-1', 'utt-2', 'utt-3']

    def test_get(self, sample_container):
        assert sample_container.get('utt-1', mem_map=False).shape == (20, 5)

    def test_remove(self, sample_container):
        sample_container.set('some-key', np.arange(20))
        assert sample_container.keys() == ['some-key', 'utt-1', 'utt-2', 'utt-3']

        sample_container.remove('some-key')
        assert sample_container.keys() == ['utt-1', 'utt-2', 'utt-3']

    def test_append(self, tmpdir):
        path = os.path.join(tmpdir.strpath, 'container')
        tmp_container = containers.Container(path)
        tmp_container.open()

        data = np.arange(100).reshape(20, 5)

        tmp_container.append('utt-1', data[:8])
        tmp_container.append('utt-1', data[8:])

        res = tmp_container.get('utt-1', mem_map=False)

        assert np.array_equal(data, res)

        tmp_container.close()

    def test_append_with_different_dimension_raises_error(self, tmpdir):
        path = os.path.join(tmpdir.strpath, 'container')
        tmp_container = containers.Container(path)
        tmp_container.open()

        tmp_container.append('utt-1', np.arange(20).reshape(5, 2, 2))

        with pytest.raises(ValueError):
            tmp_container.append('utt-1', np.arange(42).reshape(7, 2, 3))

        tmp_container.close()
