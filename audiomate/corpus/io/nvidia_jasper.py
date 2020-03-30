import os
import json
import multiprocessing

import librosa

import audiomate
from . import base

from audiomate.corpus import conversion
from audiomate import logutil

logger = logutil.getLogger()


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
        data_base_path (str): Path from where the audio files are found.
                              If ``None`` it is the path where the corpus is saved.
        no_audio_check (bool): If ``True``, the audio is not check for correct format.
        export_all_audio (bool): If ``True``, all utterances are exported,
                                 whether they are in a separate file
                                 already or not.
        transcription_label_list_idx (str): The transcriptions are used from
                                            the label-list with this id.
        sampling_rate (int): Target sampling rate to use.
        num_workers (int): Number of processes to use to process utterances.
    """

    def __init__(self, data_base_path=None,
                 no_check=False,
                 export_all_audio=False,
                 transcription_label_list_idx=audiomate.corpus.LL_WORD_TRANSCRIPT,
                 sampling_rate=16000,
                 num_workers=1):

        self.data_base_path = data_base_path
        self.no_check = no_check
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

        if self.no_check:
            out_corpus = corpus
        else:
            out_corpus = self.converter.convert(corpus, target_audio_path)

        if self.data_base_path:
            rel_base_path = self.data_base_path
        else:
            rel_base_path = path

        utts = []

        logger.info('Get utterance durations')
        utt_idx_to_path = {utt.idx: utt.track.path for utt in out_corpus.utterances.values()}
        utt_to_duration = self._get_file_durations(utt_idx_to_path)

        logger.info('Prepare utterance infos')
        for utterance in out_corpus.utterances.values():
            utt_dur = utt_to_duration[utterance.idx]
            utt_info = self.process_utterance(utterance, utt_dur, rel_base_path)
            utts.append((utterance.idx, utt_info))

        utts = sorted(utts, key=lambda x: x[1]['original_duration'])

        logger.info('Prepare subviews')
        subviews = {}
        subviews['all'] = [u[1] for u in utts]

        for subview_name, subview in corpus.subviews.items():
            utt_filter = subview.utterances.keys()
            subview_utts = [u[1] for u in utts if u[0] in utt_filter]
            subviews[subview_name] = subview_utts

        logger.info('Write files')
        for name, data in subviews.items():
            target_path = os.path.join(path, '{}.json'.format(name))
            with open(target_path, 'w') as f:
                json.dump(data, f)

    def process_utterance(self, utt, utt_dur, rel_base_path):
        rel_path = os.path.relpath(utt.track.path, rel_base_path)
        num_samples = utt_dur * self.sampling_rate

        return {
            'transcript': utt.label_lists[self.transcription_label_list_idx].join(),
            'files': [{
                'fname': rel_path,
                'channels': 1,
                'sample_rate': 16000,
                'duration': utt_dur,
                'num_samples': num_samples,
                'speed': 1
            }],
            'original_duration': utt_dur,
            'original_num_samples': num_samples,
            'utt_idx': utt.idx
        }

    def _get_file_durations(self, files):
        file_items = files.items()
        num_files = len(file_items)

        with multiprocessing.Pool(self.num_workers) as p:
            res = list(logger.progress(
                p.imap(self._get_file_duration, file_items),
                total=num_files
            ))

        return {k: v for (k, v) in res}

    def _get_file_duration(self, file_item):
        dur = librosa.core.get_duration(filename=file_item[1])
        return (file_item[0], dur)
