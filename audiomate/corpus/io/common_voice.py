import os
import glob

import audiomate
from audiomate import annotations
from audiomate import issuers
from audiomate.corpus import subset
from audiomate.utils import textfile
from . import base
from . import downloader

# DOWNLOAD_URLS taken from https://voice.mozilla.org/de/datasets
DOWNLOAD_URLS = {
    'de': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/de.tar.gz',  # noqa
    'en': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/en.tar.gz',  # noqa
    'et': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/et.tar.gz',  # noqa
    'it': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/it.tar.gz',  # noqa
    'fr': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/fr.tar.gz',  # noqa
    'cy': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/cy.tar.gz',  # noqa
    'br': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/br.tar.gz',  # noqa
    'cv': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/cv.tar.gz',  # noqa
    'tr': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/tr.tar.gz',  # noqa
    'tt': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/tt.tar.gz',  # noqa
    'ky': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/ky.tar.gz',  # noqa
    'ga-IE': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/ga-IE.tar.gz',  # noqa
    'kab': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/kab.tar.gz',  # noqa
    'ca': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/ca.tar.gz',  # noqa
    'zh-TW': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/zh-TW.tar.gz',  # noqa
    'sl': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/sl.tar.gz',  # noqa
    'nl': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/nl.tar.gz',  # noqa
    'cnh': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/cnh.tar.gz',  # noqa
    'eo': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/eo.tar.gz',  # noqa
    'fa': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/fa.tar.gz'  # noqa
}


class CommonVoiceDownloader(downloader.ArchiveDownloader):
    """
    Downloader for the Common Voice Speech Dataset.

    Args:
      lang (string): Available languages are
        de|en|et|it|fr|cy|br|cv|tr|tt|ky|ga-IE|kab|ca|zh-TW|sl|nl|cnh|eo|fa
        See Common Voice dataset Version on download page for the
        corresponding language to the shortcut
    num_threads (int): Number of threads to use for download files.
    """

    def __init__(self, lang='de', num_threads=1):
        url = DOWNLOAD_URLS[lang]

        super(CommonVoiceDownloader, self).__init__(
            url,
            move_files_up=True,
            num_threads=num_threads
        )

    @classmethod
    def type(cls):
        return 'common-voice'


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
            self.load_subset(corpus, path, subset_idx)

        return corpus

    @staticmethod
    def get_subset_ids(path):
        """ Return a list with ids of all available subsets (based on existing csv-files). """
        subset_ids = []

        for subset_path in glob.glob(os.path.join(path, '*.tsv')):
            file_name = os.path.split(subset_path)[1]
            basename = os.path.splitext(file_name)[0]

            # We don't want to include the invalidated files
            # since there maybe corrupt files
            if basename != 'invalidated':
                subset_ids.append(basename)

        return subset_ids

    def load_subset(self, corpus, path, subset_idx):
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
            file_idx = self.create_assets_if_needed(
                corpus,
                path,
                entry
            )

            if file_idx is not None:
                subset_utt_ids.append(file_idx)

        utt_filter = subset.MatchingUtteranceIdxFilter(utterance_idxs=set(subset_utt_ids))
        subview = subset.Subview(corpus, filter_criteria=[utt_filter])
        corpus.import_subview(subset_idx, subview)

    def create_assets_if_needed(self, corpus, path, entry):
        """ Create File/Utterance/Issuer, if they not already exist and return utt-idx. """
        file_name = entry[1]
        file_idx, _ = os.path.splitext(file_name)

        if file_idx in self.invalid_utterance_ids:
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
