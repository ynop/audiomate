import os

from . import base
import audiomate
from audiomate import annotations
from audiomate import issuers
from audiomate.corpus import subset
from audiomate.utils import textfile


class FluentSpeechReader(base.CorpusReader):
    """
    Reader for the Fluent Speech Commands Dataset.

    .. seealso::

       `Fluent Speech Commands Dataset <http://www.fluent.ai/research/fluent-speech-commands/>`_
          Download page
    """

    @classmethod
    def type(cls):
        return 'fluent-speech'

    def _check_for_missing_files(self, path):
        files = [
            os.path.join(path, 'data', 'speaker_demographics.csv'),
            os.path.join(path, 'data', 'train_data.csv'),
            os.path.join(path, 'data', 'test_data.csv'),
            os.path.join(path, 'data', 'valid_data.csv'),
        ]

        missing = []

        for file_path in files:
            if not os.path.isfile(file_path):
                missing.append(file_path)

        return missing

    def _load(self, path):
        corpus = audiomate.Corpus(path=path)

        speaker_path = os.path.join(path, 'data', 'speaker_demographics.csv')
        speakers = FluentSpeechReader.load_speakers(speaker_path)

        FluentSpeechReader.load_part(path, 'train', corpus, speakers)
        FluentSpeechReader.load_part(path, 'valid', corpus, speakers)
        FluentSpeechReader.load_part(path, 'test', corpus, speakers)

        return corpus

    @staticmethod
    def load_part(base_path, part_name, corpus, speakers):
        part_file_path = os.path.join(base_path, 'data', '{}_data.csv'.format(part_name))
        entries = textfile.read_separated_lines_generator(
            part_file_path,
            separator=',',
            max_columns=7,
            ignore_lines_starting_with=[',']
        )

        part_ids = []

        for entry in entries:
            file_path = entry[1]
            file_base = os.path.basename(file_path)
            idx = os.path.splitext(file_base)[0]
            speaker_idx = entry[2]
            part_ids.append(idx)

            if speaker_idx not in corpus.issuers.keys():
                corpus.import_issuers(speakers[speaker_idx])

            track = corpus.new_file(
                os.path.join(base_path, file_path),
                idx
            )

            utt = corpus.new_utterance(
                idx,
                track.idx,
                speaker_idx
            )

            transcription = annotations.LabelList.create_single(
                entry[3],
                idx=audiomate.corpus.LL_WORD_TRANSCRIPT
            )
            utt.set_label_list(transcription)

            if entry[4] != 'none':
                action = annotations.LabelList.create_single(
                    entry[4],
                    idx='action'
                )
                utt.set_label_list(action)

            if entry[5] != 'none':
                object_label = annotations.LabelList.create_single(
                    entry[5],
                    idx='object'
                )
                utt.set_label_list(object_label)

            if entry[6] != 'none':
                location = annotations.LabelList.create_single(
                    entry[6],
                    idx='location'
                )
                utt.set_label_list(location)

        utt_filter = subset.MatchingUtteranceIdxFilter(utterance_idxs=set(part_ids))
        subview = subset.Subview(corpus, filter_criteria=[utt_filter])
        corpus.import_subview(part_name, subview)

    @staticmethod
    def load_speakers(path):
        entries = textfile.read_separated_lines_generator(
            path,
            separator=',',
            max_columns=6,
            ignore_lines_starting_with=['speakerId']
        )

        idx_to_speaker = {}

        for entry in entries:
            spk = FluentSpeechReader.parse_speaker_record(entry)
            idx_to_speaker[spk.idx] = spk

        return idx_to_speaker

    @staticmethod
    def parse_speaker_record(record):
        idx = record[0]

        gender = issuers.Gender.UNKNOWN

        if record[4] == 'male':
            gender = issuers.Gender.MALE
        elif record[4] == 'female':
            gender = issuers.Gender.FEMALE

        age_group = issuers.AgeGroup.UNKNOWN

        if record[5] in ('22-40', '41-65'):
            age_group = issuers.AgeGroup.ADULT
        elif record[5] == '65+':
            age_group = issuers.AgeGroup.SENIOR

        native_lang = None

        if record[2].startswith('English'):
            native_lang = 'eng'
        elif record[2].startswith('French'):
            native_lang = 'fra'
        elif record[2].startswith('Spanish'):
            native_lang = 'spa'
        elif record[2].startswith('Telugu'):
            native_lang = 'tel'

        return issuers.Speaker(idx, gender, age_group, native_lang)
