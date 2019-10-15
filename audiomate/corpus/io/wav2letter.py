import collections
import os

import scipy
import numpy as np

import audiomate
from audiomate.utils import textfile
from . import base


class Wav2LetterWriter(base.CorpusWriter):
    """
    Writes files to use for training/testing/decoding with
    wav2letter (https://github.com/facebookresearch/wav2letter).

    Since it is expected that every utterance is in a separate file,
    any utterances that are not in separate file in the original corpus,
    are extracted into a separate file in the
    subfolder `audio` of the target path.

    Args:
        transcription_label_list_idx (str): The transcriptions are used from the label-list with this id.
    """

    def __init__(self, transcription_label_list_idx=audiomate.corpus.LL_WORD_TRANSCRIPT):
        self.transcription_label_list_idx = transcription_label_list_idx

    @classmethod
    def type(cls):
        return 'wav2letter'

    def _save(self, corpus, path):
        records = []
        subset_utterance_ids = {idx: list(subset.utterances.keys()) for idx, subset in corpus.subviews.items()}
        subset_records = collections.defaultdict(list)

        audio_folder = os.path.join(path, 'audio')
        os.makedirs(audio_folder, exist_ok=True)

        for utterance_idx in sorted(corpus.utterances.keys()):
            utterance = corpus.utterances[utterance_idx]
            export_audio = False

            if utterance.start != 0 or utterance.end != float('inf'):
                export_audio = True
            elif utterance.sampling_rate != 16000:
                # We force sr=16000, since this is expected from wav2letter
                export_audio = True

            if export_audio:
                audio_path = os.path.join(audio_folder, '{}.wav'.format(utterance.idx))
                data = utterance.read_samples(sr=16000)
                data = (data * 32768).astype(np.int16)
                num_samples = data.size
                scipy.io.wavfile.write(audio_path, 16000, data)
            else:
                audio_path = utterance.track.path
                num_samples = utterance.num_samples()

            transcript = utterance.label_lists[self.transcription_label_list_idx].join()

            # Add to the full list
            record = [utterance_idx, audio_path, num_samples, transcript]
            records.append(record)

            # Check / Add to subview lists
            for subset_idx, utt_ids in subset_utterance_ids.items():
                if utterance_idx in utt_ids:
                    subset_records[subset_idx].append(record)

        # Write full list
        records_path = os.path.join(path, 'all.lst')
        textfile.write_separated_lines(records_path, records, separator=' ', sort_by_column=-1)

        # Write subset lists
        for subset_idx, records in subset_records.items():
            if len(records) > 0:
                subset_file_path = os.path.join(path, '{}.lst'.format(subset_idx))
                textfile.write_separated_lines(subset_file_path, records, separator=' ', sort_by_column=-1)
