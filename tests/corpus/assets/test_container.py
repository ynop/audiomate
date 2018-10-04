import os

import numpy as np
from audiomate.corpus import assets

import pytest

from tests import resources


@pytest.fixture()
def sample_container():
    container = assets.Container(resources.get_resource_path(['sample_files', 'feat_container']))
    container.open()
    yield container
    container.close()


class TestContainer:

    def test_append(self, tmpdir):
        container = assets.Container(os.path.join(tmpdir.strpath, 'container'))
        container.open()

        data = np.arange(100).reshape(20, 5)

        container.append('utt-1', data[:8])
        container.append('utt-1', data[8:])

        res = container.get('utt-1', mem_map=False)

        assert np.array_equal(data, res)

        container.close()

    def test_append_with_different_dimension_raises_error(self, tmpdir):
        container = assets.Container(os.path.join(tmpdir.strpath, 'container'))
        container.open()

        container.append('utt-1', np.arange(20).reshape(5, 2, 2))

        with pytest.raises(ValueError):
            container.append('utt-1', np.arange(42).reshape(7, 2, 3))

        container.close()
