import numpy as np

from audiomate import containers
from audiomate import tracks

import pytest

from tests import resources


@pytest.fixture
def sample_container():
    container_path = resources.get_resource_path(
        ['sample_files', 'audio_container']
    )
    sample_container = containers.AudioContainer(container_path)
    sample_container.open()
    yield sample_container
    sample_container.close()


class TestContainerTrack:

    def test_sampling_rate(self, sample_container):
        sample_track = tracks.ContainerTrack('track1', sample_container)
        assert sample_track.sampling_rate == 16000

    def test_num_samples(self, sample_container):
        sample_track = tracks.ContainerTrack('track1', sample_container)
        assert sample_track.num_samples == 10

    def test_duration(self, sample_container):
        sample_track = tracks.ContainerTrack('track1', sample_container)
        assert sample_track.duration == pytest.approx(10 / 16000)

    def test_read_samples(self, sample_container):
        sample_track = tracks.ContainerTrack('track1', sample_container)
        samples = sample_track.read_samples()

        assert samples.dtype == np.float32
        assert np.allclose(
            samples,
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            atol=1.e-4
        )

    def test_read_samples_with_offset(self, sample_container):
        sample_track = tracks.ContainerTrack('track1', sample_container)
        samples = sample_track.read_samples(offset=5/16000)

        assert samples.dtype == np.float32
        assert np.allclose(
            samples,
            np.array([0.6, 0.7, 0.8, 0.9, 1.0]),
            atol=1.e-4
        )

    def test_read_samples_with_duration(self, sample_container):
        sample_track = tracks.ContainerTrack('track1', sample_container)
        samples = sample_track.read_samples(duration=5/16000)

        assert samples.dtype == np.float32
        assert np.allclose(
            samples,
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            atol=1.e-4
        )

    def test_read_samples_with_resampling(self, sample_container):
        sample_track = tracks.ContainerTrack('track1', sample_container)
        samples = sample_track.read_samples(sr=8000)

        assert samples.dtype == np.float32
        assert samples.shape[0] == 5
