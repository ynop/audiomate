import os
import glob

import audiomate
from audiomate import annotations
from audiomate import issuers
from audiomate.corpus import subset
from audiomate.utils import textfile
from . import base


INVALID_UTTS = [
    'common_voice_de_17876483',
    'common_voice_de_17876495',
    'common_voice_de_17876535',
    'common_voice_de_17876537',
    'common_voice_de_17876539',
    'common_voice_de_17876540',
    'common_voice_de_17876541',
    'common_voice_de_17876542',
    'common_voice_de_17876543',
    'common_voice_de_17876544',
    'common_voice_de_17876545',
    'common_voice_de_17876546',
    'common_voice_de_17876547',
    'common_voice_de_17876548',
    'common_voice_de_17876549',
    'common_voice_de_17876550',
    'common_voice_de_17876551',
    'common_voice_de_17876552',
    'common_voice_de_17876553',
    'common_voice_de_17876554',
    'common_voice_de_17876555',
    'common_voice_de_17876556',
    'common_voice_de_17876557',
    'common_voice_de_17876304'
]


class CommonVoiceReader(base.CorpusReader):
    """
    Reader for the Common Voice Corpus.

    .. seealso::

       `Common-Voice <https://voice.mozilla.org/>`_
          Project Page
    """

    @classmethod
    def type(cls):
        return 'common-voice'

    def _check_for_missing_files(self, path):
        return []

    def _load(self, path):
        corpus = audiomate.Corpus(path=path)
        subset_ids = CommonVoiceReader.get_subset_ids(path)

        for subset_idx in subset_ids:
            CommonVoiceReader.load_subset(corpus, path, subset_idx)

        return corpus

    @staticmethod
    def get_subset_ids(path):
        """ Return a list with ids of all available subsets (based on existing csv-files). """
        all = []

        for path in glob.glob(os.path.join(path, '*.tsv')):
            file_name = os.path.split(path)[1]
            basename = os.path.splitext(file_name)[0]

            # We don't want to include the invalidated files
            # since there maybe corrupt files
            if basename != 'invalidated':
                all.append(basename)

        return all

    @staticmethod
    def load_subset(corpus, path, subset_idx):
        """ Load subset into corpus. """
        csv_file = os.path.join(path, '{}.tsv'.format(subset_idx))
        subset_utt_ids = []

        entries = textfile.read_separated_lines_generator(
            csv_file,
            separator='\t',
            max_columns=8,
            ignore_lines_starting_with=['client_id'],
            keep_empty=True
        )

        for entry in entries:
            file_idx = CommonVoiceReader.create_assets_if_needed(
                corpus,
                path,
                entry
            )

            if file_idx is not None:
                subset_utt_ids.append(file_idx)

        filter = subset.MatchingUtteranceIdxFilter(utterance_idxs=set(subset_utt_ids))
        subview = subset.Subview(corpus, filter_criteria=[filter])
        corpus.import_subview(subset_idx, subview)

    @staticmethod
    def create_assets_if_needed(corpus, path, entry):
        """ Create File/Utterance/Issuer, if they not already exist and return utt-idx. """
        file_name = entry[1]
        file_idx, _ = os.path.splitext(file_name)

        if file_idx in INVALID_UTTS:
            return None

        if file_idx not in corpus.utterances.keys():
            speaker_idx = entry[0]
            transcription = entry[2]

            if len(entry) >= 6:
                age = CommonVoiceReader.map_age(entry[5])
            else:
                age = issuers.AgeGroup.UNKNOWN

            if len(entry) >= 7:
                gender = CommonVoiceReader.map_gender(entry[6])
            else:
                gender = issuers.Gender.UNKNOWN

            file_path = os.path.join(path, 'clips', file_name)
            corpus.new_file(file_path, file_idx)

            if speaker_idx in corpus.issuers.keys():
                issuer = corpus.issuers[speaker_idx]
            else:
                issuer = issuers.Speaker(speaker_idx, gender=gender, age_group=age)
                corpus.import_issuers(issuer)

            utterance = corpus.new_utterance(file_idx, file_idx, issuer.idx)
            utterance.set_label_list(
                annotations.LabelList.create_single(
                    transcription,
                    idx=audiomate.corpus.LL_WORD_TRANSCRIPT
                )
            )

        return file_idx

    @staticmethod
    def map_age(age):
        """ Map age to correct age-group. """

        if age in [None, '']:
            return issuers.AgeGroup.UNKNOWN
        elif age == 'teens':
            return issuers.AgeGroup.YOUTH
        elif age in ['sixties', 'seventies', 'eighties', 'nineties']:
            return issuers.AgeGroup.SENIOR
        else:
            return issuers.AgeGroup.ADULT

    @staticmethod
    def map_gender(gender):
        """ Map gender to correct value. """

        if gender == 'male':
            return issuers.Gender.MALE
        elif gender == 'female':
            return issuers.Gender.FEMALE
        else:
            return issuers.Gender.UNKNOWN
