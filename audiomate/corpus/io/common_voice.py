import glob
import os

import audiomate
import pandas as pd
from audiomate import annotations
from audiomate import issuers
from audiomate import logutil
from audiomate.corpus import subset

from . import base
from . import downloader

logger = logutil.getLogger()

# ==================================================================================================

BASE_URL = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-2020-06-22/{}.tar.gz"  # noqa
LANGUAGES = {
    "ab",
    "ar",
    "as",
    "br",
    "ca",
    "cnh",
    "cs",
    "cv",
    "cy",
    "de",
    "dv",
    "el",
    "en",
    "eo",
    "es",
    "et",
    "eu",
    "fa",
    "fr",
    "fy-NL",
    "ga-IE",
    "hsb",
    "ia",
    "id",
    "it",
    "ja",
    "ka",
    "kab",
    "ky",
    "lv",
    "mn",
    "mt",
    "nl",
    "or",
    "pa-IN",
    "pl",
    "pt",
    "rm-sursilv",
    "rm-vallader",
    "ro",
    "ru",
    "rw",
    "sah",
    "sl",
    "sv-SE",
    "ta",
    "tr",
    "tt",
    "uk",
    "vi",
    "vot",
    "zh-CN",
    "zh-HK",
    "zh-TW",
}


# ==================================================================================================


class CommonVoiceDownloader(downloader.ArchiveDownloader):
    """
    Downloader for the Common Voice Speech Dataset.

    Args:
      lang (string): See Common Voice dataset Version on download page for the
        corresponding language to the shortcut
    num_threads (int): Number of threads to use for download files.
    """

    def __init__(self, lang="de"):
        if lang in LANGUAGES:
            link = BASE_URL.format(lang)
            super(CommonVoiceDownloader, self).__init__(
                link, move_files_up=True, num_threads=1
            )
        else:
            msg = "There is no common-voice URL present for language {}!"
            raise ValueError(msg.format(lang))

    @classmethod
    def type(cls):
        return "common-voice"


# ==================================================================================================


class CommonVoiceReader(base.CorpusReader):
    """
    Reader for the Common Voice Corpus.

    .. seealso::

       `Common-Voice <https://voice.mozilla.org/>`_
          Project Page
    """

    @classmethod
    def type(cls):
        return "common-voice"

    def _check_for_missing_files(self, path):
        return []

    # ==============================================================================================

    def _load(self, path):
        corpus = audiomate.Corpus(path=path)
        subset_ids = CommonVoiceReader.get_subset_ids(path)

        for subset_idx in subset_ids:
            self.load_subset(corpus, path, subset_idx)

        return corpus

    # ==============================================================================================

    @staticmethod
    def get_subset_ids(path):
        """ Return a list with ids of all available subsets (based on existing tsv-files) """

        subset_ids = []
        for subset_path in glob.glob(os.path.join(path, "*.tsv")):
            file_name = os.path.split(subset_path)[1]
            basename = os.path.splitext(file_name)[0]

            # We don't want to include the invalidated files since there maybe corrupt files
            if basename != "invalidated" and basename != "reported":
                subset_ids.append(basename)

        return subset_ids

    # ==============================================================================================

    def load_subset(self, corpus, path, subset_idx):
        """ Load subset into corpus """

        tsv_file = os.path.join(path, "{}.tsv".format(subset_idx))

        data = pd.read_csv(tsv_file, keep_default_na=False, sep="\t")
        file_ids = data.apply(
            lambda row: self.create_assets_if_needed(corpus, path, row), axis=1
        )
        file_ids = [idx for idx in file_ids if idx is not None]

        utt_filter = subset.MatchingUtteranceIdxFilter(utterance_idxs=set(file_ids))
        subview = subset.Subview(corpus, filter_criteria=[utt_filter])
        corpus.import_subview(subset_idx, subview)

    # ==============================================================================================

    def create_assets_if_needed(self, corpus, path, entry):
        """ Create File/Utterance/Issuer, if they not already exist and return utt-idx. """

        file_name = entry["path"]
        file_idx, _ = os.path.splitext(file_name)

        if file_idx in self.invalid_utterance_ids:
            return None

        if file_idx not in corpus.utterances.keys():
            speaker_idx = entry["client_id"]
            transcription = entry["sentence"]

            age = CommonVoiceReader.map_age(entry["age"])
            gender = CommonVoiceReader.map_gender(entry["gender"])

            file_path = os.path.join(path, "clips", file_name)
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
                    transcription, idx=audiomate.corpus.LL_WORD_TRANSCRIPT
                )
            )

        return file_idx

    # ==============================================================================================

    @staticmethod
    def map_age(age):
        """ Map age to correct age-group. """

        if age in [None, ""]:
            return issuers.AgeGroup.UNKNOWN
        elif age == "teens":
            return issuers.AgeGroup.YOUTH
        elif age in ["sixties", "seventies", "eighties", "nineties"]:
            return issuers.AgeGroup.SENIOR
        else:
            return issuers.AgeGroup.ADULT

    # ==============================================================================================

    @staticmethod
    def map_gender(gender):
        """ Map gender to correct value. """

        if gender == "male":
            return issuers.Gender.MALE
        elif gender == "female":
            return issuers.Gender.FEMALE
        else:
            return issuers.Gender.UNKNOWN
