import os
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

    def test_read_frames(self, tmpdir):
        cont_path = os.path.join(tmpdir.strpath, 'audio.hdf5')
        cont = containers.AudioContainer(cont_path)
        cont.open()

        content = np.random.random(10044)
        cont.set('track', content, 16000)
        track = tracks.ContainerTrack('some_idx', cont, 'track')

        data = list(track.read_frames(frame_size=400, hop_size=160))
        frames = np.array([x[0] for x in data])
        last = [x[1] for x in data]

        assert frames.shape == (62, 400)
        assert frames.dtype == np.float32
        assert np.allclose(frames[0], content[:400], atol=0.0001)
        expect = np.pad(content[9760:], (0, 116), mode='constant')
        assert np.allclose(frames[61], expect, atol=0.0001)

        assert last[:-1] == [False] * (len(data) - 1)
        assert last[-1]

        cont.close()

    def test_read_frames_matches_length(self, tmpdir):
        cont_path = os.path.join(tmpdir.strpath, 'audio.hdf5')
        cont = containers.AudioContainer(cont_path)
        cont.open()

        content = np.random.random(7)
        cont.set('track', content, 16000)
        track = tracks.ContainerTrack('some_idx', cont, 'track')

        data = list(track.read_frames(frame_size=2, hop_size=1))
        frames = np.array([x[0] for x in data])
        last = [x[1] for x in data]

        assert frames.shape == (6, 2)
        assert frames.dtype == np.float32

        assert last[:-1] == [False] * (len(data) - 1)
        assert last[-1]

        cont.close()
