import os
import collections

import numpy as np
import scipy

import audiomate
from . import base
from audiomate.utils import textfile


class MozillaDeepSpeechWriter(base.CorpusWriter):
    """
    Writes files to use for training with Mozilla DeepSpeech (https://github.com/mozilla/DeepSpeech).

    Since it is expected that every utterance is in a separate file,
    any utterances that are not in separate file in the original corpus,
    are extracted into a separate file in the subfolder `audio` of the target path.

    Args:
        transcription_label_list_idx (str): The transcriptions are used from the label-list with this id.
    """

    def __init__(self, transcription_label_list_idx=audiomate.corpus.LL_WORD_TRANSCRIPT):
        self.transcription_label_list_idx = transcription_label_list_idx

    @classmethod
    def type(cls):
        return 'mozilla-deepspeech'

    def _save(self, corpus, path):
        records = []
        subset_utterance_ids = {idx: list(subset.utterances.keys()) for idx, subset in corpus.subviews.items()}
        subset_records = collections.defaultdict(list)

        audio_folder = os.path.join(path, 'audio')
        os.makedirs(audio_folder, exist_ok=True)

        for utterance_idx in sorted(corpus.utterances.keys()):
            utterance = corpus.utterances[utterance_idx]

            if utterance.start == 0 and utterance.end == float('inf'):
                audio_path = utterance.track.path
            else:
                audio_path = os.path.join(audio_folder, '{}.wav'.format(utterance.idx))
                sampling_rate = utterance.sampling_rate
                data = utterance.read_samples()

                data = (data * 32768).astype(np.int16)

                scipy.io.wavfile.write(audio_path, sampling_rate, data)

            size = os.stat(audio_path).st_size
            transcript = utterance.label_lists[self.transcription_label_list_idx].join()

            # Add to the full list
            record = [audio_path, size, transcript]
            records.append(record)

            # Check / Add to subview lists
            for subset_idx, utt_ids in subset_utterance_ids.items():
                if utterance_idx in utt_ids:
                    subset_records[subset_idx].append(record)

        # Write full list
        records.insert(0, ['wav_filename', 'wav_filesize', 'transcript'])
        records_path = os.path.join(path, 'all.csv')
        textfile.write_separated_lines(records_path, records, separator=',', sort_by_column=-1)

        # Write subset lists
        for subset_idx, records in subset_records.items():
            if len(records) > 0:
                records.insert(0, ['wav_filename', 'wav_filesize', 'transcript'])
                subset_file_path = os.path.join(path, '{}.csv'.format(subset_idx))
                textfile.write_separated_lines(subset_file_path, records, separator=',', sort_by_column=-1)
