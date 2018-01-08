"""
This module contains the base classes for processors.
A processor is used to process the samples of utterances from a corpus.

There are different levels of abstractions for a processor.
On top there is the ``:py:class:`pingu.corpus.preprocessing.Processor`, which provides the basic structure to process
the utterances one after another.

Based on that there are two subclasses ``:py:class:`pingu.corpus.preprocessing.OnlineProcessor` and
``:py:class:`pingu.corpus.preprocessing.OfflineProcessor`. The first one is streaming-processor for processing samples
frame by frame without the need for loading the full utterance.

The latter one is used to process all samples of the utterance at once.
In a lot of cases this is easier to implement.
"""

import abc
import math

import librosa
import numpy as np

from pingu.corpus import assets


class Processor(metaclass=abc.ABCMeta):
    """
    This class is the base class for all kind of feature extraction.
    The processor produces from a given corpus features, which it then stores in a feature-container.

    For implementing a specific processor, the ``process_utterance`` method has to be implemented:
        * This method is called for every utterance in the corpus.
        * In the method any feature extraction / pre-processing can be done.
        * The result then has to be saved in the feature-container, which is passed along with the utterance.
          The result has to be saved, with the id of the utterance, which is passed as argument.
    """

    def process_corpus(self, corpus, output_path, frame_size=400, hop_size=160):
        """
        Process the given corpus and save the processed features in a feature-container at the given path.

        Args:
            corpus (Corpus): The corpus to process the utterances from.
            output_path (str): A path to save the feature-container to.
            frame_size (int): The number of samples per frame.
            hop_size (int): The number of samples between two frames.

        Returns:
            FeatureContainer: The feature-container containing the processed features.
        """

        feat_container = assets.FeatureContainer(output_path)
        feat_container.open()

        sampling_rate = -1

        for utterance in corpus.utterances.values():
            utt_sampling_rate = utterance.sampling_rate

            if sampling_rate > 0 and sampling_rate != utt_sampling_rate:
                raise ValueError(
                    'File {} has a different sampling-rate than the previous ones!'.format(utterance.file.idx))

            sampling_rate = utt_sampling_rate

            self.process_utterance(utterance, feat_container, corpus=corpus, frame_size=frame_size, hop_size=hop_size)

        feat_container.frame_size = frame_size
        feat_container.hop_size = hop_size
        feat_container.sampling_rate = sampling_rate

        feat_container.close()

        return feat_container

    @abc.abstractmethod
    def process_utterance(self, utterance, feature_container, corpus=None, frame_size=400, hop_size=160):
        """
        Extract features of the given utterances and put it in the given feature-container.

        Args:
            utterance (Utterance): The utterance to process.
            feature_container (FeatureContainer): The feature-container to store the output.
            corpus (Corpus): The corpus where the utterance is from, if available.
            frame_size (int): The number of samples per frame.
            hop_size (int): The number of samples between two frames.
        """
        pass


class OfflineProcessor(Processor, metaclass=abc.ABCMeta):
    """
    This class should be used for feature extraction in batch mode (one full utterance in a step).

    For implementing a specific offline-processor, the ``process_sequence`` method has to be implemented:
        * As input the method receives a 2-Dimensional array of frames (n-frames x n-samples-per-frame).
        * It must return a array with the first dimension of the same size as the input.

    Note:
        The samples are padded with zeros to match the number of frames equal to
        math.ceil((20 - self.frame_size) / self.hop_size + 1).

    """

    def process_utterance(self, utterance, feature_container, corpus=None, frame_size=400, hop_size=160):
        samples = utterance.read_samples()

        if samples.size <= 0:
            raise ValueError('Utterance {} has no samples'.format(utterance.idx))

        # Pad with zeros to match frames
        num_frames = math.ceil((samples.size - frame_size) / hop_size + 1)
        num_pad_samples = (num_frames - 1) * hop_size + frame_size

        if num_pad_samples > samples.size:
            samples = np.pad(samples, (0, num_pad_samples - samples.size), mode='constant', constant_values=0)

        frames = librosa.util.frame(samples, frame_length=frame_size, hop_length=hop_size).T
        processed = self.process_sequence(frames, utterance.sampling_rate, utterance=utterance, corpus=corpus)
        feature_container.set(utterance.idx, processed)

    @abc.abstractmethod
    def process_sequence(self, frames, sampling_rate, utterance=None, corpus=None):
        """
        Process the given frames, which represent an utterance.

        Args:
            frames (numpy.ndarray): (n-frames x n-samples-per-frame) Frames.
            sampling_rate (int): The sampling rate of the underlying signal.
            corpus (Corpus): The corpus where the data is from, if available.
            utterance (Utterance): The utterance the data is from, if available.

        Returns:
            numpy.ndarray: (n-frames x ...) The features extracted from the given samples.
        """
        pass
