import numpy as np

from . import base
from pingu.utils import units


class FrameClassificationGrabber(base.Grabber):
    """
    The FrameClassificationGrabber provides access to the data, so it can be used for classification of single frames.
    The returned sample contains the data of one frame and the corresponding labels encoded as binary vector.

    Args:
        label_list_idx (str): A single label-list-identifier.
                              Only labels from this given label-list will be included in the label vector.
        label_values (list): A list of label values. Only these labels will be included in the encoded label vector.
                             Every label vector will contain one value for each label in this list in the same order.
                             If None, all available labels will be used.

    Notes:
        Since the frames can be grabbed randomly, the pre-processing-pipeline can not be used for tasks,
        where the output depends on other frames.

    """

    def __init__(self, corpus, feature_container, label_list_idx=None, label_values=None):
        super(FrameClassificationGrabber, self).__init__(corpus, feature_container)

        self.label_list_idx = label_list_idx

        if label_values is None:
            self.label_values = self._extract_labels()
        else:
            self.label_values = label_values

        self.frame_size = self.feature_container.frame_size
        self.hop_size = self.feature_container.hop_size
        self.sampling_rate = self.feature_container.sampling_rate

        self.segments = self._extract_segments()

    def __len__(self):
        last = self.segments[-1]
        return last[1] + last[2]

    def __getitem__(self, item):
        segment = self._segment_for_index(item)
        local_index = item - segment[1] + segment[3]
        utterance_idx = segment[0]

        frame = self.feature_container.get(utterance_idx)[local_index].copy()

        return frame, segment[4]

    def _segment_for_index(self, index):
        for segment in self.segments:
            if segment[1] <= index < segment[1] + segment[2]:
                return segment

    def _extract_labels(self):
        """
        Get a sorted list of all label values occurring in the label-lists given by `self.label_lists`.
        """
        all_labels = self.corpus.all_label_values(self.label_list_idx)
        return sorted(all_labels)

    def _extract_segments(self):
        """
        Get all segments of the data set. A segment is continuous sequence of frames with the same label vector.
        """
        segments = []
        global_frame_offset = 0

        for utterance_idx, utterance in sorted(self.corpus.utterances.items(), key=lambda x: x[0]):

            if self.label_list_idx in utterance.label_lists.keys():
                # Get the corresponding label-list
                label_list = utterance.label_lists[self.label_list_idx]
                num_frames = self.feature_container.get(utterance_idx).shape[0]

                ranges = label_list.ranges(include_labels=self.label_values)

                for range_start, range_end, labels in ranges:
                    abs_start = range_start
                    first_frame_index = units.seconds_to_frame(abs_start, self.hop_size, self.sampling_rate)

                    if range_end < 0:
                        last_frame_index = num_frames - 1
                    else:
                        abs_end = range_end
                        abs_end_sample = units.seconds_to_sample(abs_end, self.sampling_rate)
                        last_frame_index = units.sample_to_frame(abs_end_sample - 1, self.hop_size)

                    label_vec = self._label_vec_from_labels(labels)

                    frames_in_segment = (last_frame_index - first_frame_index + 1)
                    segments.append((
                        utterance_idx,
                        global_frame_offset,
                        frames_in_segment,
                        first_frame_index,
                        label_vec
                    ))

                    global_frame_offset += frames_in_segment

        return segments

    def _label_vec_from_labels(self, labels):
        """
        Create an encoded label vector from the given labels.

        ('a', 'b', 'd') --> [1,1,0,1]
        """
        vec = np.zeros(len(self.label_values)).astype(np.float32)

        for label in labels:
            index = self.label_values.index(label.value)
            vec[index] = 1

        return vec
