import os

import audiomate
from audiomate import annotations
from audiomate import issuers
from audiomate import logutil
from audiomate.corpus import subset
from audiomate.utils import textfile

from . import base
from . import downloader

logger = logutil.getLogger()

# ==================================================================================================

BASE_URL = "https://www.caito.de/data/Training/stt_tts/{}.tgz"
LANGUAGES = {
    "de_DE",
    "en_UK",
    "en_US",
    "es_ES",
    "fr_FR",
    "it_IT",
    "pl_PL",
    "ru_RU",
    "uk_UK",
}


# ==================================================================================================


class MailabsDownloader(downloader.ArchiveDownloader):
    """
    Downloader for the M-AILABS Speech Dataset.

    """

    def __init__(self, lang="de_DE"):
        if lang in LANGUAGES:
            link = BASE_URL.format(lang)
            super(MailabsDownloader, self).__init__(
                link, move_files_up=True, num_threads=1
            )
        else:
            msg = "There is no mailabs URL present for language {}!"
            raise ValueError(msg.format(lang))

    @classmethod
    def type(cls):
        return "mailabs"


# ==================================================================================================


class MailabsReader(base.CorpusReader):
    """
    Reader for the M-AILABS Speech Dataset.

    .. seealso::

       `M-AILABS Speech Dataset <http://www.m-ailabs.bayern/en/the-mailabs-speech-dataset/>`_
          Project Page
    """

    @classmethod
    def type(cls):
        return "mailabs"

    def _check_for_missing_files(self, path):
        return []

    # ==============================================================================================

    def _load(self, path):
        corpus = audiomate.Corpus(path=path)

        items = os.listdir(path)
        while len(items) == 1:
            # The directories have a slightly different structure in different languages, so go
            #  through the subfolders until the current directory has more than one single folder
            path = os.path.join(path, items[0])
            items = os.listdir(path)

        tag_folders = MailabsReader.get_folders(path)
        for tag_folder in tag_folders:
            self.load_tag(corpus, tag_folder)

        return corpus

    # ==============================================================================================

    @staticmethod
    def get_folders(path):
        """ Return a list of all subfolder-paths in the given path. """

        folder_paths = []
        items = os.listdir(path)

        for item in items:
            folder_path = os.path.join(path, item)

            if os.path.isdir(folder_path):
                folder_paths.append(folder_path)

        return folder_paths

    # ==============================================================================================

    def load_tag(self, corpus, path):
        """
        Iterate over all speakers on load them.
        Collect all utterance-idx and create a subset of them.
        """

        tag_idx = os.path.basename(path)
        logger.info("Loading speakers - {}".format(tag_idx))
        speaker_books = []

        if tag_idx == "mix":
            # IN MIX FOLDERS THERE ARE NO SPEAKERS
            # HANDLE EVERY UTT AS DIFFERENT ISSUER
            speaker_books.append([path, None])
        else:
            for speaker_path in MailabsReader.get_folders(path):
                speaker = MailabsReader.load_speaker(corpus, speaker_path)
                speaker_books.append([speaker_path, speaker])

        tag_utt_ids = []
        for book, speaker in speaker_books:
            utt_ids = self.load_books_of_speaker(corpus, book, speaker)
            tag_utt_ids.extend(utt_ids)

        utt_filter = subset.MatchingUtteranceIdxFilter(utterance_idxs=set(tag_utt_ids))
        subview = subset.Subview(corpus, filter_criteria=[utt_filter])
        corpus.import_subview(tag_idx, subview)

    # ==============================================================================================

    @staticmethod
    def load_speaker(corpus, path):
        """ Create a speaker instance for the given path.  """
        base_path, speaker_name = os.path.split(path)
        base_path, gender_desc = os.path.split(base_path)
        base_path, _ = os.path.split(base_path)
        base_path, _ = os.path.split(base_path)

        gender = issuers.Gender.UNKNOWN

        if gender_desc == "male":
            gender = issuers.Gender.MALE
        elif gender_desc == "female":
            gender = issuers.Gender.FEMALE

        speaker = issuers.Speaker(speaker_name, gender=gender)
        corpus.import_issuers(speaker)

        return speaker

    # ==============================================================================================

    def load_books_of_speaker(self, corpus, path, speaker):
        """ Load all utterances for the speaker at the given path. """
        utt_ids = []

        for book_path in MailabsReader.get_folders(path):
            meta_path = os.path.join(book_path, "metadata.csv")
            wavs_path = os.path.join(book_path, "wavs")

            meta = textfile.read_separated_lines(
                meta_path, separator="|", max_columns=3
            )

            for entry in meta:
                file_basename = entry[0]
                transcription_raw = entry[1]
                transcription_clean = entry[2]

                if speaker is None:
                    idx = file_basename
                    utt_speaker = issuers.Speaker(idx)
                    speaker_idx = idx
                    corpus.import_issuers(utt_speaker)
                else:
                    idx = "{}-{}".format(speaker.idx, file_basename)
                    speaker_idx = speaker.idx

                wav_name = "{}.wav".format(file_basename)
                wav_path = os.path.join(wavs_path, wav_name)

                if os.path.isfile(wav_path) and idx not in self.invalid_utterance_ids:
                    corpus.new_file(wav_path, idx)

                    ll_raw = annotations.LabelList.create_single(
                        transcription_raw, idx=audiomate.corpus.LL_WORD_TRANSCRIPT_RAW
                    )

                    ll_clean = annotations.LabelList.create_single(
                        transcription_clean, idx=audiomate.corpus.LL_WORD_TRANSCRIPT
                    )

                    utterance = corpus.new_utterance(idx, idx, speaker_idx)
                    utterance.set_label_list(ll_raw)
                    utterance.set_label_list(ll_clean)

                    utt_ids.append(utterance.idx)

        return utt_ids
