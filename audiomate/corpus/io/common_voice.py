import os
import glob

import audiomate
from audiomate import annotations
from audiomate import issuers
from audiomate import logutil
from audiomate.corpus import subset
from audiomate.utils import textfile
from . import base
from . import downloader

logger = logutil.getLogger()

# DOWNLOAD_URLS taken from https://voice.mozilla.org/de/datasets
DOWNLOAD_URLS = {
    'ab': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/ab.tar.gz',  # noqa
    'ar': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/ar.tar.gz',  # noqa
    'as': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/as.tar.gz',  # noqa
    'br': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/br.tar.gz',  # noqa
    'ca': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/ca.tar.gz',  # noqa
    'cnh': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/cnh.tar.gz',  # noqa
    'cs': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/cs.tar.gz',  # noqa
    'cv': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/cv.tar.gz',  # noqa
    'cy': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/cy.tar.gz',  # noqa
    'de': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/de.tar.gz',  # noqa
    'dv': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/dv.tar.gz',  # noqa
    'el': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/el.tar.gz',  # noqa
    'en': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/en.tar.gz',  # noqa
    'eo': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/eo.tar.gz',  # noqa
    'es': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/es.tar.gz',  # noqa
    'et': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/et.tar.gz',  # noqa
    'eu': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/eu.tar.gz',  # noqa
    'fa': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/fa.tar.gz',  # noqa
    'fr': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/fr.tar.gz',  # noqa
    'fy-NL': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/fy-NL.tar.gz',  # noqa
    'ga-IE': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/ga-IE.tar.gz',  # noqa
    'hsb': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/hsb.tar.gz',  # noqa
    'ia': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/ia.tar.gz',  # noqa
    'id': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/id.tar.gz',  # noqa
    'it': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/it.tar.gz',  # noqa
    'ja': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/ja.tar.gz',  # noqa
    'ka': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/ka.tar.gz',  # noqa
    'kab': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/kab.tar.gz',  # noqa
    'ky': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/ky.tar.gz',  # noqa
    'lv': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/lv.tar.gz',  # noqa
    'mn': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/mn.tar.gz',  # noqa
    'mt': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/mt.tar.gz',  # noqa
    'nl': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/nl.tar.gz',  # noqa
    'or': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/or.tar.gz',  # noqa
    'pa-IN': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/pa-IN.tar.gz',  # noqa
    'pl': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/pl.tar.gz',  # noqa
    'pt': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/pt.tar.gz',  # noqa
    'rm-sursilv': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/rm-sursilv.tar.gz',  # noqa
    'rm-vallader': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/rm-vallader.tar.gz',  # noqa
    'ro': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/ro.tar.gz',  # noqa
    'ru': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/ru.tar.gz',  # noqa
    'rw': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/rw.tar.gz',  # noqa
    'sah': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/sah.tar.gz',  # noqa
    'sl': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/sl.tar.gz',  # noqa
    'sv-SE': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/sv-SE.tar.gz',  # noqa
    'ta': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/ta.tar.gz',  # noqa
    'tr': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/tr.tar.gz',  # noqa
    'tt': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/tt.tar.gz',  # noqa
    'uk': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/uk.tar.gz',  # noqa
    'vi': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/vi.tar.gz',  # noqa
    'vot': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/vot.tar.gz',  # noqa
    'zh-CN': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/zh-CN.tar.gz',  # noqa
    'zh-HK': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/zh-TW.tar.gz',  # noqa
    'zh-TW': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/zh-TW.tar.gz'  # noqa
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
            ignore_lines_starting_with=['accent'],
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
        file_name = entry[6]
        file_idx, _ = os.path.splitext(file_name)

        if file_idx in self.invalid_utterance_ids:
            return None

        if file_idx not in corpus.utterances.keys():
            speaker_idx = entry[2]
            transcription = entry[7]

            age = CommonVoiceReader.map_age(entry[1])
            gender = CommonVoiceReader.map_gender(entry[4])

            file_path = os.path.join(path, 'clips', file_name)
            if not os.path.exists(file_path):
                # In some languages (like german) the audio files are in a 'clips' subfolder,
                #  in others (like french or spanish) they are saved in the main directory,
                #  so try both possibilities
                file_path = os.path.join(path, file_name)
                if not os.path.exists(file_path):
                    msg = "Skipping file because it doesn't exist: {}"
                    logger.warn(msg.format(file_path))
                    return None

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
