import math

import numpy as np
from scipy.io import wavfile

from pingu.utils import units


class FramedSignalGrabber(object):
    """
    The FramedSignalGrabber provides access to the samples of the audio data in frames.
    The whole dataset is split into ranges. One range is a section with the same labels.
    Only ranges are considered that have labels from the given label-list. Ranges are padded with zeros, if the frame-length doesn't match.

    Args:
        corpus (Corpus): The corpus to get the data from.
        label_list_idx (str): Only label-lists with this id will be considered.
        frame_length (int): Number of audio samples per frame.
        hop_size (int): Number of audio samples from one to the next frame.
        include_labels (list): If not empty, only the label values in the list will be grabbed.
        predefined_labels (list): If not empty, this is used as output structure. Only the given labels will be included.

    Attributes:
        ranges (list): List of all ranges (frame_offset, num_frames, start_sample, num_samples, file_data, label_vec).
        labels (list): List of all labels occurring in the output of the grabber.
    """

    def __init__(self, corpus, label_list_idx='default', frame_length=400, hop_size=160, include_labels=None, predefined_labels=None):
        self.corpus = corpus
        self.label_list_idx = label_list_idx
        self.frame_length = frame_length
        self.hop_size = hop_size
        self.include_labels = include_labels

        if predefined_labels is not None:
            self.labels = list(predefined_labels)
            self.include_labels = self.labels
        else:
            self.labels = self._extract_labels()

        self.ranges = self._extract_ranges()

    def __len__(self):
        last = self.ranges[-1]
        return last[0] + last[1]

    def __getitem__(self, item):
        range = self._range_for_index(item)
        frame_index = item - range[0]
        sample_start = range[2] + self.hop_size * frame_index
        sample_end = range[2] + range[3]
        sample_pad = 0

        if sample_start + self.frame_length <= sample_end:
            sample_end = sample_start + self.frame_length
        else:
            sample_pad = sample_start + self.frame_length - sample_end

        frame = np.array(range[4][sample_start:sample_end])
        max_value = np.iinfo(frame.dtype).max

        frame = frame.astype(np.float) / max_value

        if sample_pad:
            frame = np.pad(frame, (0, sample_pad), mode='constant', constant_values=0)

        return frame, range[5]

    def _range_for_index(self, index):
        for range in self.ranges:
            if range[0] <= index < range[0] + range[1]:
                return range

    def _extract_labels(self):
        label_lists = self.corpus.label_lists[self.label_list_idx]
        all_labels = set([])

        for utterance_idx, label_list in label_lists.items():
            all_labels.update(label_list.label_values())

        if self.include_labels is not None:
            all_labels = all_labels.intersection(set(self.include_labels))

        return sorted(all_labels)

    def _extract_ranges(self):
        """
        Get all ranges of the dataset. Range --> (frame_offset, num_frames, start_sample, num_samples, file_data, label_vec)
        """
        ranges = []
        files = {}
        offset = 0

        label_lists = self.corpus.label_lists[self.label_list_idx]

        for utterance in self.corpus.utterances.values():
            if utterance.idx in label_lists.keys():

                # Get matrix with audio data of the file that contains the utterance
                if utterance.file_idx in files.keys():
                    sample_rate, file_data = files[utterance.file_idx]
                else:
                    file = self.corpus.files[utterance.file_idx]
                    sample_rate, file_data = wavfile.read(file.path, mmap=True)
                    files[utterance.file_idx] = (sample_rate, file_data)

                # Get the corresponding label-list
                label_list = label_lists[utterance.idx]

                # Extract ranges
                utt_ranges, offset = self._extract_ranges_from_utterance(utterance, file_data, label_list, offset, sample_rate)
                ranges.extend(utt_ranges)

        return ranges

    def _extract_ranges_from_utterance(self, utterance, file_data, label_list, frame_offset, sample_rate):
        """
        Extract the ranges from the given utterance.

        Returns:
            (list, int): List of ranges (frame_offset, num_frames, start_sample, num_samples, file_data, label_vec), new frame offset
        """
        ranges = []
        offset = frame_offset

        for (range_start, range_end, labels) in label_list.ranges(include_labels=self.include_labels):
            start, length = self._range_start_len_relative_to_file_in_samples(utterance, file_data, range_start, range_end, sample_rate)
            num_frames = self._frames_from_length(length)
            label_vec = self._label_vec_from_labels(labels)

            ranges.append((offset, num_frames, start, length, file_data, label_vec))

            offset += num_frames

        return ranges, offset

    def _range_start_len_relative_to_file_in_samples(self, utterance, file_data, range_start, range_end, sample_rate):
        """
        Calculate the range start and length in samples relative to the audio file.
        """
        utt_start = units.seconds_to_sample(utterance.start, sampling_rate=sample_rate)
        utt_len = file_data.size - utt_start

        if utterance.end > 0:
            utt_len = units.seconds_to_sample(utterance.end, sampling_rate=sample_rate) - utt_start

        start = utt_start + units.seconds_to_sample(range_start, sampling_rate=sample_rate)
        length = utt_len - start

        if range_end > -1:
            length = utt_start + units.seconds_to_sample(range_end, sampling_rate=sample_rate) - start

        return start, length

    def _frames_from_length(self, length):
        """
        Calculate the number of frames that are generated from the given number of samples.
        """
        return math.ceil(length / self.hop_size)

    def _label_vec_from_labels(self, labels):
        """
        Create an encoded label vector from the given labels.

        ('a', 'b', 'd') --> [1,1,0,1]
        """
        vec = np.zeros(len(self.labels)).astype(np.float)

        for label in labels:
            index = self.labels.index(label.value)
            vec[index] = 1

        return vec
