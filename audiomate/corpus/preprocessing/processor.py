import abc

import librosa
import numpy as np

from audiomate.corpus import assets
from audiomate.utils import units


class Processor(metaclass=abc.ABCMeta):
    """
    This class is the base class for all kind of feature extraction.
    The processor produces from a given corpus features, which it then stores in a feature-container.

    For implementing a specific processor, the ``process_utterance`` method has to be implemented:

        * This method is called for every utterance in the corpus.
        * In the method any feature extraction / pre-processing can be done.
        * The result then has to be saved in the feature-container, which is passed along with the utterance.
          The result has to be saved, with the id of the utterance, which is passed as argument.

    Example:
        >>> import audiomate
        >>> from audiomate.corpus.preprocessing.pipeline import offline
        >>>
        >>> ds = audiomate.Corpus.load('some/corpus/path')
        >>> mfcc_processor = offline.MFCC(n_mfcc=13, n_mels=128)
        >>> norm_processor = offline.MeanVarianceNorm(mean=5.4, variance=2.3, parent=mfcc_processor)
        >>>
        >>> fc = norm_processor.process_corpus(ds, output_path='path/mfcc_features.h5', frame_size=400, hop_size=160)
        >>> fc
        <audiomate.corpus.assets.features.FeatureContainer at 0x10d451a20>
        >>> fc.open()
        >>> fc.get('existing-utterance-id')[()]
        array([[-6.18418212,  3.93379946,  2.51237535,  3.62199459, -6.77845303,
         3.28746939,  1.36316432, -0.7814685 , -2.36003147,  3.27370797,
        -3.24373709, -2.42513017, -1.55695699],
        ...
    """

    def process_corpus(self, corpus, output_path, frame_size=400, hop_size=160, sr=None):
        """
        Process the given corpus and save the processed features in a feature-container at the given path.

        Args:
            corpus (Corpus): The corpus to process the utterances from.
            output_path (str): A path to save the feature-container to.
            frame_size (int): The number of samples per frame.
            hop_size (int): The number of samples between two frames.
            sr (int): Use the given sampling rate. If None uses the native sampling rate from the file.

        Returns:
            FeatureContainer: The feature-container containing the processed features.
        """

        feat_container = assets.FeatureContainer(output_path)
        feat_container.open()

        sampling_rate = -1

        for utterance in corpus.utterances.values():
            utt_sampling_rate = utterance.sampling_rate

            if sr is None:
                if sampling_rate > 0 and sampling_rate != utt_sampling_rate:
                    raise ValueError(
                        'File {} has a different sampling-rate than the previous ones!'.format(utterance.file.idx))

                sampling_rate = utt_sampling_rate

            self.process_utterance(utterance, feat_container,
                                   corpus=corpus,
                                   frame_size=frame_size,
                                   hop_size=hop_size,
                                   sr=sr)

        feat_container.frame_size = frame_size
        feat_container.hop_size = hop_size
        feat_container.sampling_rate = sr or sampling_rate

        feat_container.close()

        return feat_container

    def process_corpus_from_feature_container(self, corpus, input_features, output_path):
        """
        Process the given corpus and save the processed features in a feature-container at the given path.
        Instead of using the framed signal, use the features from from the given feature-container.

        Args:
            corpus (Corpus): The corpus to process the utterances from.
            input_features (FeatureContainer): The feature-container to process the frames from.
            output_path (str): A path to save the feature-container to.

        Returns:
            FeatureContainer: The feature-container containing the processed features.
        """

        feat_container = assets.FeatureContainer(output_path)
        feat_container.open()

        input_features.open()

        for utterance in corpus.utterances.values():
            self.process_utterance_from_feature_container(utterance, input_features, feat_container, corpus)

        feat_container.frame_size = input_features.frame_size
        feat_container.hop_size = input_features.hop_size
        feat_container.sampling_rate = input_features.sampling_rate

        feat_container.close()

        return feat_container

    @abc.abstractmethod
    def process_utterance(self, utterance, feature_container, corpus=None, frame_size=400, hop_size=160, sr=None):
        """
        Extract features of the given utterances and put it in the given feature-container.

        Args:
            utterance (Utterance): The utterance to process.
            feature_container (FeatureContainer): The feature-container to store the output.
            corpus (Corpus): The corpus where the utterance is from, if available.
            frame_size (int): The number of samples per frame.
            hop_size (int): The number of samples between two frames.
            sr (int): Use the given sampling rate. If None uses the native sampling rate from the file.
        """
        pass

    @abc.abstractmethod
    def process_utterance_from_feature_container(self, utterance, in_feat_container, out_feat_container, corpus=None):
        """
        Process the feature of the given utterance from the given input feature-container and put it
        to the given output feature-container.

        Args:
            utterance (Utterance): The utterance to process.
            in_feat_container (FeatureContainer): The feature-container to read the input frames.
            out_feat_container (FeatureContainer): The feature-container to store the output.
            corpus (Corpus): The corpus where the utterance is from, if available.
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
        math.ceil((num_samples - self.frame_size) / self.hop_size + 1).

    """

    def process_utterance(self, utterance, feature_container, corpus=None, frame_size=400, hop_size=160, sr=None):
        frame_settings = units.FrameSettings(frame_size, hop_size)
        samples = utterance.read_samples(sr=sr)

        if samples.size <= 0:
            raise ValueError('Utterance {} has no samples'.format(utterance.idx))

        # Pad with zeros to match frames
        num_frames = frame_settings.num_frames(samples.size)
        num_pad_samples = (num_frames - 1) * hop_size + frame_size

        if num_pad_samples > samples.size:
            samples = np.pad(samples, (0, num_pad_samples - samples.size), mode='constant', constant_values=0)

        # Get sampling-rate if not given
        sampling_rate = sr or utterance.sampling_rate

        frames = librosa.util.frame(samples, frame_length=frame_size, hop_length=hop_size).T
        processed = self.process_sequence(frames, sampling_rate, utterance=utterance, corpus=corpus)
        feature_container.set(utterance.idx, processed)

    def process_utterance_from_feature_container(self, utterance, in_feat_container, out_feat_container, corpus=None):
        sampling_rate = in_feat_container.sampling_rate
        frames = in_feat_container.get(utterance.idx, mem_map=False)
        processed = self.process_sequence(frames, sampling_rate, utterance=utterance, corpus=corpus)
        out_feat_container.set(utterance.idx, processed)

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
