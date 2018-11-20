import os
import numpy as np

from audiomate import containers

import pytest
from tests import resources


@pytest.fixture()
def sample_container():
    container_path = resources.get_resource_path(
        ['sample_files', 'audio_container']
    )
    sample_container = containers.AudioContainer(container_path)
    sample_container.open()
    yield sample_container
    sample_container.close()


class TestAudioContainer:

    def test_get(self, sample_container):
        samples, sr = sample_container.get('track1')

        assert np.allclose(
            samples,
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            atol=1.e-4
        )
        assert sr == 16000

    def test_set(self, tmpdir):
        path = os.path.join(tmpdir.strpath, 'audio')
        cnt = containers.AudioContainer(path)
        cnt.open()

        data = np.random.random(10)
        cnt.set('track1', data, 16000)

        assert cnt.keys() == ['track1']

        samples, sr = cnt.get('track1')

        assert np.allclose(samples, data, atol=1.e-4)
        assert sr == 16000

        cnt.close()

    def test_set_raises_error_on_invalid_data_type(self, tmpdir):
        path = os.path.join(tmpdir.strpath, 'audio')
        cnt = containers.AudioContainer(path)
        cnt.open()

        with pytest.raises(ValueError):
            cnt.set('track1', np.arange(5).astype(np.int16), 16000)

        cnt.close()

    def test_set_raises_error_on_multi_dim_data(self, tmpdir):
        path = os.path.join(tmpdir.strpath, 'audio')
        cnt = containers.AudioContainer(path)
        cnt.open()

        with pytest.raises(ValueError):
            cnt.set('track1',
                    np.random.random((4, 5)).astype(np.float32),
                    16000)

        cnt.close()

    def test_append(self, tmpdir):
        path = os.path.join(tmpdir.strpath, 'audio')
        cnt = containers.AudioContainer(path)
        cnt.open()

        chunk_a = np.random.random(10).astype(np.float32)
        cnt.append('track1', np.array(chunk_a), 16000)
        samples, sr = cnt.get('track1')

        assert np.allclose(samples, chunk_a, atol=1.e-4)
        assert sr == 16000

        chunk_b = np.random.random(5).astype(np.float32)
        cnt.append('track1', np.array(chunk_b), 16000)
        samples, sr = cnt.get('track1')

        assert np.allclose(
            samples,
            np.concatenate([chunk_a, chunk_b]),
            atol=1.e-4
        )
        assert sr == 16000

        cnt.close()
