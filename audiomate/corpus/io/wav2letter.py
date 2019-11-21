import collections
import os

import audiomate
from audiomate.utils import textfile
from . import base
from audiomate.corpus import conversion


class Wav2LetterWriter(base.CorpusWriter):
    """
    Writes files to use for training/testing/decoding with
    wav2letter (https://github.com/facebookresearch/wav2letter).

    Since it is expected that every utterance is in a separate file,
    any utterances that are not in separate file in the original corpus,
    are extracted into a separate file in the
    subfolder `audio` of the target path.

    Args:
        export_all_audio (bool): If ``True``, all utterances are exported,
                                 whether they are in a separate file
                                 already or not.
        transcription_label_list_idx (str): The transcriptions are used from the label-list with this id.
        num_workers (int): Number of processes to use to process utterances.
    """

    def __init__(self, export_all_audio=False,
                 transcription_label_list_idx=audiomate.corpus.LL_WORD_TRANSCRIPT,
                 num_workers=1):
        self.export_all_audio = export_all_audio
        self.transcription_label_list_idx = transcription_label_list_idx
        self.sampling_rate = 16000
        self.num_workers = num_workers

        self.converter = conversion.WavAudioFileConverter(
            self.num_workers,
            self.sampling_rate,
            separate_file_per_utterance=True,
            force_conversion=self.export_all_audio
        )

    @classmethod
    def type(cls):
        return 'wav2letter'

    def _save(self, corpus, path):
        target_audio_path = os.path.join(path, 'audio')
        os.makedirs(target_audio_path, exist_ok=True)

        # Convert all files
        corpus = self.converter.convert(corpus, target_audio_path)
        records = []

        subset_utterance_ids = {idx: list(subset.utterances.keys()) for idx, subset in corpus.subviews.items()}
        subset_records = collections.defaultdict(list)

        for utterance_idx in sorted(corpus.utterances.keys()):
            utterance = corpus.utterances[utterance_idx]
            transcript = utterance.label_lists[self.transcription_label_list_idx].join()
            audio_path = utterance.track.path
            num_samples = int(utterance.duration * self.sampling_rate)

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
