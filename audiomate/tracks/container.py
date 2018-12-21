import copy

import librosa
import numpy as np

from . import track


class ContainerTrack(track.Track):
    """
    A track that is stored in a :py:class:`audiomate.containers.AudioContainer`.

    Args:
        idx (str): A identifier to uniquely identify a track.
        container (AudioContainer): The audio container with the samples.
        key (str): The key of the samples in the container.
                   If ``None``, it is assumed it's the same
                   as ``idx``.
    """
    __slots__ = ['container', 'key']

    def __init__(self, idx, container, key=None):
        super(ContainerTrack, self).__init__(idx)

        self.container = container

        if key is None:
            self.key = idx
        else:
            self.key = key

    def __copy__(self):
        return ContainerTrack(
            self.idx,
            self.container,
            key=self.key
        )

    def __deepcopy(self, memo):
        return ContainerTrack(
            self.idx,
            copy.deepcopy(self.container, memo),
            key=self.key
        )

    @property
    def sampling_rate(self):
        """
        Return the sampling rate.
        """
        with self.container.open_if_needed(mode='r') as cnt:
            return cnt.get(self.key)[1]

    @property
    def num_channels(self):
        """
        Return the number of channels.
        """
        return 1

    @property
    def num_samples(self):
        """
        Return the total number of samples.
        """
        with self.container.open_if_needed(mode='r') as cnt:
            return cnt.get(self.key)[0].shape[0]

    @property
    def duration(self):
        """
        Return the duration in seconds.
        """
        with self.container.open_if_needed(mode='r') as cnt:
            samples, sr = cnt.get(self.key)

            return samples.shape[0] / sr

    def read_samples(self, sr=None, offset=0, duration=None):
        """
        Return the samples from the track in the container.
        Uses librosa for resampling, if needed.

        Args:
            sr (int): If ``None``, uses the sampling rate given by the file,
                      otherwise resamples to the given sampling rate.
            offset (float): The time in seconds, from where to start reading
                            the samples (rel. to the file start).
            duration (float): The length of the samples to read in seconds.

        Returns:
            np.ndarray: A numpy array containing the samples as a
            floating point (numpy.float32) time series.
        """
        with self.container.open_if_needed(mode='r') as cnt:
            samples, native_sr = cnt.get(self.key)

            start_sample_index = int(offset * native_sr)

            if duration is None:
                end_sample_index = samples.shape[0]
            else:
                end_sample_index = int((offset + duration) * native_sr)

            samples = samples[start_sample_index:end_sample_index]

            if sr is not None and sr != native_sr:
                samples = librosa.core.resample(
                    samples,
                    native_sr,
                    sr,
                    res_type='kaiser_best'
                )

            return samples

    def read_frames(self, frame_size, hop_size, offset=0,
                    duration=None, block_size=None):
        """
        Generator that reads and returns the samples of the track in frames.

        Args:
            frame_size (int): The number of samples per frame.
            hop_size (int): The number of samples between two frames.
            offset (float): The time in seconds, from where to start
                            reading the samples (rel. to the track start).
            duration (float): The length of the samples to read in seconds.

        Returns:
            Generator: A generator yielding a tuple for every frame.
            The first item is the frame,
            the second the sampling-rate and
            the third a boolean indicating if it is the last frame.
        """
        with self.container.open_if_needed(mode='r') as cnt:
            samples, sr = cnt.get(self.key)

            current_index = 0

            while current_index + frame_size < samples.shape[0]:
                next_frame = samples[current_index:current_index+frame_size]
                yield next_frame, False
                current_index += hop_size

            next_frame = samples[current_index:]

            if next_frame.shape[0] < frame_size:
                next_frame = np.pad(
                    next_frame,
                    (0, frame_size - next_frame.shape[0]),
                    mode='constant',
                    constant_values=0
                )

            yield next_frame, True
