import os
import collections

import audiomate
from . import base
from audiomate.utils import textfile
from audiomate.corpus import conversion


class MozillaDeepSpeechWriter(base.CorpusWriter):
    """
    Writes files to use for training with Mozilla DeepSpeech (https://github.com/mozilla/DeepSpeech).

    Since it is expected that every utterance is in a separate file,
    any utterances that are not in separate file in the original corpus,
    are extracted into a separate file in the subfolder `audio` of the target path.

    Args:
        no_audio_check (bool): If ``True``, the audio is not checked for correct format.
        export_all_audio (bool): If ``True``, all utterances are exported,
                                 whether they are in a separate file
                                 already or not.
        transcription_label_list_idx (str): The transcriptions are used from the label-list with this id.
        sampling_rate (int): Target sampling rate to use.
        num_workers (int): Number of processes to use to process utterances.
    """

    def __init__(self, export_all_audio=False,
                 no_audio_check=False,
                 transcription_label_list_idx=audiomate.corpus.LL_WORD_TRANSCRIPT,
                 sampling_rate=16000, num_workers=1):
        self.no_audio_check = no_audio_check
        self.export_all_audio = export_all_audio
        self.transcription_label_list_idx = transcription_label_list_idx
        self.sampling_rate = sampling_rate
        self.num_workers = num_workers

        self.converter = conversion.WavAudioFileConverter(
            self.num_workers,
            self.sampling_rate,
            separate_file_per_utterance=True,
            force_conversion=self.export_all_audio
        )

    @classmethod
    def type(cls):
        return 'mozilla-deepspeech'

    def _save(self, corpus, path):
        target_audio_path = os.path.join(path, 'audio')
        os.makedirs(target_audio_path, exist_ok=True)

        # Convert all files
        if not self.no_audio_check:
            corpus = self.converter.convert(corpus, target_audio_path)

        records = []

        subset_utterance_ids = {idx: set(subset.utterances.keys()) for idx, subset in corpus.subviews.items()}
        subset_records = collections.defaultdict(list)

        for utterance_idx in sorted(corpus.utterances.keys()):
            utterance = corpus.utterances[utterance_idx]
            transcript = utterance.label_lists[self.transcription_label_list_idx].join()
            audio_path = utterance.track.path
            size = os.stat(audio_path).st_size

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
