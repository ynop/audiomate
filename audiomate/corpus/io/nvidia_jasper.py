import os
import json

import audiomate
from . import base

from audiomate.corpus import conversion


class NvidiaJasperWriter(base.CorpusWriter):
    """
    Writes files to use for training with NVIDIA Jasper
    (https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper).

    **Audio Files:**
    Since utterances are expected to be in a separate file and have
    a specific format (WAV 16-Bit PCM), any utterances that do not meet those
    requirements, are extracted into a separate file in the
    subfolder `audio` of the target path.

    **Subviews:**
    Subviews in the Jasper format are represented as different
    Json-Files. For every subview in the corpus a separate
    Json-File is created. Additionaly, a ``all.json`` is created,
    that contains all utterances.

    Args:
        export_all_audio (bool): If ``True``, all utterances are exported,
                                 whether they are in a separate file
                                 already or not.
        transcription_label_list_idx (str): The transcriptions are used from
                                            the label-list with this id.
        sampling_rate (int): Target sampling rate to use.
        num_workers (int): Number of processes to use to process utterances.
    """

    def __init__(self, export_all_audio=False,
                 transcription_label_list_idx=audiomate.corpus.LL_WORD_TRANSCRIPT,
                 sampling_rate=16000,
                 num_workers=1):

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
        return 'nvidia-jasper'

    def _save(self, corpus, path):
        target_audio_path = os.path.join(path, 'audio')
        os.makedirs(target_audio_path, exist_ok=True)

        out_corpus = self.converter.convert(corpus, target_audio_path)

        utts = []

        for utterance in out_corpus.utterances.values():
            utt_info = self.process_utterance(utterance, path)
            utts.append((utterance.idx, utt_info))

        utts = sorted(utts, key=lambda x: x[1]['original_duration'])

        print('subviews')
        subviews = {}
        subviews['all'] = [u[1] for u in utts]

        for subview_name, subview in corpus.subviews.items():
            utt_filter = subview.utterances.keys()
            subview_utts = [u[1] for u in utts if u[0] in utt_filter]
            subviews[subview_name] = subview_utts

        print('write')
        for name, data in subviews.items():
            target_path = os.path.join(path, '{}.json'.format(name))
            with open(target_path, 'w') as f:
                json.dump(data, f)

    def process_utterance(self, utt, base_path):
        rel_path = os.path.relpath(utt.track.path, base_path)
        duration = utt.duration
        num_samples = utt.duration * self.sampling_rate

        return {
            'transcript': utt.label_lists[self.transcription_label_list_idx].join(),
            'files': [{
                'fname': rel_path,
                'channels': 1,
                'sample_rate': 16000,
                'duration': duration,
                'num_samples': num_samples,
                'speed': 1
            }],
            'original_duration': duration,
            'original_num_samples': num_samples
        }
