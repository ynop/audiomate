import abc

import librosa
import numpy as np

from audiomate import containers
from audiomate.utils import units


class Processor(metaclass=abc.ABCMeta):
    """
    The processor base class provides the functionality to process audio data on different levels
    (Corpus, Utterance, Track). For every level there is an offline and an online method.
    In the offline mode the data is processed in one step (e.g. the whole track/utterance at once).
    This means the ``process_frames`` method is called with all the frames of the track/utterance.
    In online mode the data is processed in chunks, so the ``process_frames`` method is called multiple times
    per track/utterance with different chunks.

    To implement a concrete processor the ``process_frames`` method has to be implemented.
    This method is called in online and offline mode. So it is up to the user to determine
    if a processor can be called in either online or offline mode, maybe both. This differs between use cases.

    If the implementation of a processor does change the frame or hop-size,
    it is expected to provide a transform via the ``frame_transform`` method.
    Frame-size and hop-size are measured in samples regarding the original audio signal (or simply its sampling rate).
    """

    def process_corpus(self, corpus, output_path, frame_size=400, hop_size=160, sr=None):
        """
        Process all utterances of the given corpus and save the processed features in a feature-container.
        The utterances are processed in **offline** mode so the full utterance in one go.

        Args:
            corpus (Corpus): The corpus to process the utterances from.
            output_path (str): A path to save the feature-container to.
            frame_size (int): The number of samples per frame.
            hop_size (int): The number of samples between two frames.
            sr (int): Use the given sampling rate. If None uses the native sampling rate from the underlying data.

        Returns:
            FeatureContainer: The feature-container containing the processed features.
        """

        def processing_func(utterance, feat_container, frame_size, hop_size, sr, corpus):
            data = self.process_utterance(utterance, frame_size=frame_size, hop_size=hop_size, sr=sr, corpus=corpus)
            feat_container.set(utterance.idx, data)

        return self._process_corpus(corpus, output_path, processing_func,
                                    frame_size=frame_size, hop_size=hop_size, sr=sr)

    def process_corpus_online(self, corpus, output_path, frame_size=400, hop_size=160,
                              chunk_size=1, buffer_size=5760000):
        """
        Process all utterances of the given corpus and save the processed features in a feature-container.
        The utterances are processed in **online** mode, so chunk by chunk.

        Args:
            corpus (Corpus): The corpus to process the utterances from.
            output_path (str): A path to save the feature-container to.
            frame_size (int): The number of samples per frame.
            hop_size (int): The number of samples between two frames.
            chunk_size (int): Number of frames to process per chunk.
            buffer_size (int): Number of samples to load into memory at once.
                             The exact number of loaded samples depends on the block-size of the audioread library.
                             So it can be of block-size higher, where the block-size is typically 1024 or 4096.

        Returns:
            FeatureContainer: The feature-container containing the processed features.
        """

        def processing_func(utterance, feat_container, frame_size, hop_size, corpus, sr):
            for chunk in self.process_utterance_online(utterance,
                                                       frame_size=frame_size,
                                                       hop_size=hop_size,
                                                       corpus=corpus,
                                                       chunk_size=chunk_size,
                                                       buffer_size=buffer_size):
                feat_container.append(utterance.idx, chunk)

        return self._process_corpus(corpus, output_path, processing_func,
                                    frame_size=frame_size, hop_size=hop_size, sr=None)

    def process_features(self, corpus, input_features, output_path):
        """
        Process all features of the given corpus and save the processed features in a feature-container.
        The features are processed in **offline** mode, all features of an utterance at once.

        Args:
            corpus (Corpus): The corpus to process the utterances from.
            input_features (FeatureContainer): The feature-container to process the frames from.
            output_path (str): A path to save the feature-container to.

        Returns:
            FeatureContainer: The feature-container containing the processed features.
        """
        feat_container = containers.FeatureContainer(output_path)
        feat_container.open()

        input_features.open()

        for utterance in corpus.utterances.values():
            sampling_rate = input_features.sampling_rate
            frames = input_features.get(utterance.idx, mem_map=False)
            processed = self.process_frames(frames, sampling_rate, offset=0, last=True,
                                            utterance=utterance, corpus=corpus)
            feat_container.set(utterance.idx, processed)

        tf_frame_size, tf_hop_size = self.frame_transform(input_features.frame_size, input_features.hop_size)
        feat_container.frame_size = tf_frame_size
        feat_container.hop_size = tf_hop_size
        feat_container.sampling_rate = input_features.sampling_rate

        feat_container.close()

        return feat_container

    def process_features_online(self, corpus, input_features, output_path, chunk_size=1):
        """
        Process all features of the given corpus and save the processed features in a feature-container.
        The features are processed in **online** mode, chunk by chunk.

        Args:
            corpus (Corpus): The corpus to process the utterances from.
            input_features (FeatureContainer): The feature-container to process the frames from.
            output_path (str): A path to save the feature-container to.
            chunk_size (int): Number of frames to process per chunk.

        Returns:
            FeatureContainer: The feature-container containing the processed features.
        """
        feat_container = containers.FeatureContainer(output_path)
        feat_container.open()

        input_features.open()

        for utterance in corpus.utterances.values():
            sampling_rate = input_features.sampling_rate
            frames = input_features.get(utterance.idx, mem_map=True)

            current_frame = 0

            while current_frame < frames.shape[0]:
                last = current_frame + chunk_size > frames.shape[0]
                to_frame = current_frame + chunk_size

                chunk = frames[current_frame:to_frame]

                processed = self.process_frames(chunk, sampling_rate, current_frame,
                                                last=last, utterance=utterance, corpus=corpus)

                if processed is not None:
                    feat_container.append(utterance.idx, processed)

                current_frame += chunk_size

        tf_frame_size, tf_hop_size = self.frame_transform(input_features.frame_size, input_features.hop_size)
        feat_container.frame_size = tf_frame_size
        feat_container.hop_size = tf_hop_size
        feat_container.sampling_rate = input_features.sampling_rate

        feat_container.close()

        return feat_container

    def process_utterance(self, utterance, frame_size=400, hop_size=160, sr=None, corpus=None):
        """
        Process the utterance in **offline** mode, in one go.

        Args:
            utterance (Utterance): The utterance to process.
            frame_size (int): The number of samples per frame.
            hop_size (int): The number of samples between two frames.
            sr (int): Use the given sampling rate. If None uses the native sampling rate from the underlying data.
            corpus (Corpus): The corpus this utterance is part of, if available.

        Returns:
            np.ndarray: The processed features.
        """
        return self.process_track(utterance.track, frame_size=frame_size, hop_size=hop_size, sr=sr,
                                  start=utterance.start, end=utterance.end, utterance=utterance, corpus=corpus)

    def process_utterance_online(self, utterance, frame_size=400, hop_size=160, chunk_size=1,
                                 buffer_size=5760000, corpus=None):
        """
        Process the utterance in **online** mode, chunk by chunk.
        The processed chunks are yielded one after another.

        Args:
            utterance (Utterance): The utterance to process.
            frame_size (int): The number of samples per frame.
            hop_size (int): The number of samples between two frames.
            chunk_size (int): Number of frames to process per chunk.
            buffer_size (int): Number of samples to load into memory at once.
                             The exact number of loaded samples depends on the block-size of the audioread library.
                             So it can be of block-size higher, where the block-size is typically 1024 or 4096.
            corpus (Corpus): The corpus this utterance is part of, if available.

        Returns:
            Generator: A generator that yield processed chunks.
        """
        return self.process_track_online(utterance.track,
                                         frame_size=frame_size,
                                         hop_size=hop_size,
                                         start=utterance.start,
                                         end=utterance.end,
                                         utterance=utterance,
                                         corpus=corpus,
                                         chunk_size=chunk_size,
                                         buffer_size=buffer_size)

    def process_track(self, track, frame_size=400, hop_size=160, sr=None,
                      start=0, end=float('inf'), utterance=None, corpus=None):
        """
        Process the track in **offline** mode, in one go.

        Args:
            track (Track): The track to process.
            frame_size (int): The number of samples per frame.
            hop_size (int): The number of samples between two frames.
            sr (int): Use the given sampling rate. If ``None``,
                      uses the native sampling rate from the underlying data.
            start (float): The point within the track in seconds,
                           to start processing from.
            end (float): The point within the track in seconds,
                         to end processing.
            utterance (Utterance): The utterance that is associated with
                                   this track, if available.
            corpus (Corpus): The corpus this track is part of, if available.

        Returns:
            np.ndarray: The processed features.
        """
        frame_settings = units.FrameSettings(frame_size, hop_size)

        if end != float('inf'):
            samples = track.read_samples(sr=sr, offset=start, duration=end-start)
        else:
            samples = track.read_samples(sr=sr, offset=start)

        if sr is None:
            sr = track.sampling_rate

        if samples.size <= 0:
            raise ValueError('Track {} has no samples'.format(track.idx))

        # Pad with zeros to match frames
        num_frames = frame_settings.num_frames(samples.size)
        num_pad_samples = (num_frames - 1) * hop_size + frame_size

        if num_pad_samples > samples.size:
            samples = np.pad(samples, (0, num_pad_samples - samples.size), mode='constant', constant_values=0)

        # Get sampling-rate if not given
        sampling_rate = sr or utterance.sampling_rate

        frames = librosa.util.frame(samples, frame_length=frame_size, hop_length=hop_size).T
        return self.process_frames(frames, sampling_rate, 0, last=True, utterance=utterance, corpus=corpus)

    def process_track_online(self, track, frame_size=400, hop_size=160,
                             start=0, end=float('inf'), utterance=None, corpus=None,
                             chunk_size=1, buffer_size=5760000):
        """
        Process the track in **online** mode, chunk by chunk.
        The processed chunks are yielded one after another.

        Args:
            track (Track): The track to process.
            frame_size (int): The number of samples per frame.
            hop_size (int): The number of samples between two frames.
            start (float): The point within the track in seconds to start processing from.
            end (float): The point within the trac in seconds to end processing.
            utterance (Utterance): The utterance that is associated with this track, if available.
            corpus (Corpus): The corpus this track is part of, if available.
            chunk_size (int): Number of frames to process per chunk.
            buffer_size (int): Number of samples to load into memory at once.
                               The exact number of loaded samples depends
                               on the type of track.
                               It can be of block-size higher,
                               where the block-size is typically 1024 or 4096.

        Returns:
            Generator: A generator that yield processed chunks.
        """

        current_frame = 0
        frames = []

        duration = None
        sr = track.sampling_rate

        if end != float('inf'):
            duration = end - start

        # Process chunks that are within end bounds
        for frame, is_last in track.read_frames(frame_size,
                                                hop_size,
                                                offset=start,
                                                duration=duration,
                                                buffer_size=buffer_size):

            frames.append(frame)

            if len(frames) == chunk_size:
                processed = self.process_frames(np.array(frames), sr, current_frame,
                                                last=is_last, utterance=utterance, corpus=corpus)
                if processed is not None:
                    yield processed
                current_frame += chunk_size
                frames = frames[chunk_size:]

        # Process overlapping chunks with zero frames at the end
        if len(frames) > 0:
            processed = self.process_frames(np.array(frames), sr, current_frame,
                                            last=True, utterance=utterance, corpus=corpus)
            yield processed

    @abc.abstractmethod
    def process_frames(self, data, sampling_rate, offset=0, last=False, utterance=None, corpus=None):
        """
        Process the given chunk of frames. Depending on online or offline mode,
        the given chunk is either the full data or just part of it.

        Args:
            data (np.ndarray): nD Array of frames (num-frames x frame-dimensions).
            sampling_rate (int): The sampling rate of the underlying signal.
            offset (int): The index of the first frame in the chunk. In offline mode always 0.
                          (Relative to the first frame of the utterance/sequence)
            last (bool): True indicates that this is the last frame of the sequence/utterance.
                         In offline mode always True.
            utterance (Utterance): The utterance the frame is from, if available.
            corpus (Corpus): The corpus the frame is from, if available.

        Returns:
            np.ndarray: The processed frames.
        """
        pass

    def frame_transform(self, frame_size, hop_size):
        """
        If the processor changes the number of samples that build up a frame or
        the number of samples between two consecutive frames (hop-size),
        this function needs transform the original frame- and/or hop-size.

        This is used to store the frame-size and hop-size in a feature-container.
        In the end one can calculate start and end time of a frame with this information.

        By default it is assumed that the processor doesn't change the frame-size and the hop-size.

        Args:
            frame_size (int): The original frame-size.
            hop_size (int): The original hop-size.

        Returns:
            tuple: The (frame-size, hop-size) after processing.
        """
        return frame_size, hop_size

    def _process_corpus(self, corpus, output_path, processing_func, frame_size=400, hop_size=160, sr=None):
        """ Utility function for processing a corpus with a separate processing function. """
        feat_container = containers.FeatureContainer(output_path)
        feat_container.open()

        sampling_rate = -1

        for utterance in corpus.utterances.values():
            utt_sampling_rate = utterance.sampling_rate

            if sr is None:
                if sampling_rate > 0 and sampling_rate != utt_sampling_rate:
                    raise ValueError(
                        'File {} has a different sampling-rate than the previous ones!'.format(utterance.track.idx))

                sampling_rate = utt_sampling_rate

            processing_func(utterance, feat_container, frame_size, hop_size, sr, corpus)

        tf_frame_size, tf_hop_size = self.frame_transform(frame_size, hop_size)
        feat_container.frame_size = tf_frame_size
        feat_container.hop_size = tf_hop_size
        feat_container.sampling_rate = sr or sampling_rate

        feat_container.close()

        return feat_container
