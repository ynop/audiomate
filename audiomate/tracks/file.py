import copy

import librosa
import audioread

from . import track
from audiomate.utils import audio


class FileTrack(track.Track):
    """
    A track that is stored in a file.

    Args:
        idx (str): A identifier to uniquely identify a track.
        path (str): The path to the file.
    """
    __slots__ = ['path']

    def __init__(self, idx, path):
        super(FileTrack, self).__init__(idx)
        self.path = path

    def __copy__(self):
        return FileTrack(self.idx, self.path)

    def __deepcopy(self, memo):
        return copy.copy(self)

    @property
    def sampling_rate(self):
        """
        Return the sampling rate.
        """
        with audioread.audio_open(self.path) as f:
            return f.samplerate

    @property
    def num_channels(self):
        """
        Return the number of channels.
        """
        with audioread.audio_open(self.path) as f:
            return f.channels

    @property
    def num_samples(self):
        """
        Return the total number of samples.
        """
        with audioread.audio_open(self.path) as f:
            return int(f.duration * f.samplerate)

    @property
    def duration(self):
        """
        Return the duration in seconds.
        """
        with audioread.audio_open(self.path) as f:
            return f.duration

    def read_samples(self, sr=None, offset=0, duration=None):
        """
        Return the samples from the file.
        Uses librosa for loading
        (see http://librosa.github.io/librosa/generated/librosa.core.load.html).

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
        samples, __ = librosa.core.load(
            self.path,
            sr=sr,
            offset=offset,
            duration=duration
        )
        return samples

    def read_frames(self, frame_size, hop_size, offset=0,
                    duration=None, buffer_size=5760000):
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
        if duration is not None:
            end = offset + duration
        else:
            end = float('inf')

        return audio.read_frames(
            self.path,
            frame_size,
            hop_size,
            start=offset,
            end=end,
            buffer_size=buffer_size)
