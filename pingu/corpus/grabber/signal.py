import numpy as np
from scipy.io import wavfile

from pingu.utils import units


class FramedSignalGrabber(object):
    def __init__(self, corpus, label_list_idx='default', frame_length=400, hop_size=160, sampling_rate=22050):
        self.corpus = corpus
        self.label_list_idx = label_list_idx
        self.frame_length = frame_length
        self.hop_size = hop_size
        self.sampling_rate = 22050

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def _extract_ranges(self):
        ranges = []
        files = {}

        label_lists = self.corpus.label_lists[self.label_list_idx]

        for utterance in self.corpus.utterances.values():
            if utterance.file_idx in files.keys():
                file_data = files[utterance.file_idx]
            else:
                file = self.corpus.files[utterance.file_idx]
                file_data = wavfile.read(file.path, mmap=True)
                files[utterance.file_idx] = file_data

            utt_start = units.seconds_to_sample(utterance.start, sampling_rate=self.sampling_rate)
            utt_len = file_data.size - utt_start

            if utterance.end > 0:
                utt_len = units.seconds_to_sample(utterance.end, sampling_rate=self.sampling_rate) - utt_start

            if utterance.idx in label_lists.keys():
                label_list = label_lists[utterance.idx]

                for (range_start, range_end, labels) in label_list.ranges():
                    range_start = units.seconds_to_sample(range_start, sampling_rate=self.sampling_rate)
                    range_len = utt_len - range_start

                    if range_end > -1:
                        range_len = units.seconds_to_sample(range_end, sampling_rate=self.sampling_rate) - range_start

                    range_offset = utt_start + range_start

                    ranges.append((range_offset, range_len, file_data, labels))

        return ranges
