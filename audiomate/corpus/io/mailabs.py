import os

import audiomate
from audiomate import annotations
from audiomate import issuers
from audiomate.corpus import subset
from audiomate.utils import download
from audiomate.utils import textfile

from . import base

DOWNLOAD_URLS = {
    'de_DE': 'https://www.caito.de/data/Training/stt_tts/de_DE.tgz',
    'en_UK': 'https://www.caito.de/data/Training/stt_tts/en_UK.tgz',
    'en_US': 'https://www.caito.de/data/Training/stt_tts/en_US.tgz',
    'es_ES': 'https://www.caito.de/data/Training/stt_tts/es_ES.tgz',
    'it_IT': 'https://www.caito.de/data/Training/stt_tts/it_IT.tgz',
    'uk_UK': 'https://www.caito.de/data/Training/stt_tts/uk_UK.tgz',
    'ru_RU': 'https://www.caito.de/data/Training/stt_tts/ru_RU.tgz',
    'fr_FR': 'https://www.caito.de/data/Training/stt_tts/fr_FR.tgz',
    'pl_PL': 'https://www.caito.de/data/Training/stt_tts/pl_PL.tgz',
}


class MailabsDownloader(base.CorpusDownloader):
    """
    Downloader for the M-AILABS Speech Dataset.

    Args:
        tags (list): List of tags for different parts to download.
                     Corresponds to the tags in the
                     `Statistics & Download Links` on the webpage.
                     If ``None``, all parts are downloaded.
        num_threads (int): Number of threads to use for download files.
    """

    def __init__(self, tags=None, num_threads=1):
        self.tags = tags
        self.num_threads = num_threads

    @classmethod
    def type(cls):
        return 'mailabs'

    def _download(self, target_path):
        os.makedirs(target_path, exist_ok=True)

        for tag, download_url in DOWNLOAD_URLS.items():
            if self.tags is None or tag in self.tags:
                tmp_file = os.path.join(target_path, 'tmp_{}.tgz'.format(tag))

                download.download_file(
                    download_url,
                    tmp_file,
                    num_threads=self.num_threads
                )
                download.extract_tar(tmp_file, target_path)

                os.remove(tmp_file)


class MailabsReader(base.CorpusReader):
    """
    Reader for the M-AILABS Speech Dataset.

    .. seealso::

       `M-AILABS Speech Dataset <http://www.m-ailabs.bayern/en/the-mailabs-speech-dataset/>`_
          Project Page
    """

    @classmethod
    def type(cls):
        return 'mailabs'

    def _check_for_missing_files(self, path):
        return []

    def _load(self, path):
        corpus = audiomate.Corpus(path=path)
        tag_folders = MailabsReader.get_folders(path)

        for tag_folder in tag_folders:
            self.load_tag(corpus, tag_folder)

        return corpus

    @staticmethod
    def get_folders(path):
        """ Return a list of all subfolder-paths in the given path. """
        folder_paths = []

        for item in os.listdir(path):
            folder_path = os.path.join(path, item)

            if os.path.isdir(folder_path):
                folder_paths.append(folder_path)

        return folder_paths

    def load_tag(self, corpus, path):
        """
        Iterate over all speakers on load them.
        Collect all utterance-idx and create a subset of them.
        """
        tag_idx = os.path.basename(path)
        data_path = os.path.join(path, 'by_book')
        tag_utt_ids = []

        for gender_path in MailabsReader.get_folders(data_path):
            # IN MIX FOLDERS THERE ARE NO SPEAKERS
            # HANDLE EVERY UTT AS DIFFERENT ISSUER
            if os.path.basename(gender_path) == 'mix':
                utt_ids = self.load_books_of_speaker(corpus,
                                                     gender_path,
                                                     None)
                tag_utt_ids.extend(utt_ids)

            else:
                for speaker_path in MailabsReader.get_folders(gender_path):
                    speaker = MailabsReader.load_speaker(corpus, speaker_path)
                    utt_ids = self.load_books_of_speaker(corpus,
                                                         speaker_path,
                                                         speaker)

                    tag_utt_ids.extend(utt_ids)

        utt_filter = subset.MatchingUtteranceIdxFilter(
            utterance_idxs=set(tag_utt_ids)
        )
        subview = subset.Subview(corpus, filter_criteria=[utt_filter])
        corpus.import_subview(tag_idx, subview)

    @staticmethod
    def load_speaker(corpus, path):
        """ Create a speaker instance for the given path.  """
        base_path, speaker_name = os.path.split(path)
        base_path, gender_desc = os.path.split(base_path)
        base_path, _ = os.path.split(base_path)
        base_path, _ = os.path.split(base_path)

        gender = issuers.Gender.UNKNOWN

        if gender_desc == 'male':
            gender = issuers.Gender.MALE
        elif gender_desc == 'female':
            gender = issuers.Gender.FEMALE

        speaker = issuers.Speaker(speaker_name, gender=gender)
        corpus.import_issuers(speaker)

        return speaker

    def load_books_of_speaker(self, corpus, path, speaker):
        """ Load all utterances for the speaker at the given path. """
        utt_ids = []

        for book_path in MailabsReader.get_folders(path):
            meta_path = os.path.join(book_path, 'metadata.csv')
            wavs_path = os.path.join(book_path, 'wavs')

            meta = textfile.read_separated_lines(meta_path,
                                                 separator='|',
                                                 max_columns=3)

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
                    idx = '{}-{}'.format(speaker.idx, file_basename)
                    speaker_idx = speaker.idx

                wav_name = '{}.wav'.format(file_basename)
                wav_path = os.path.join(wavs_path, wav_name)

                if os.path.isfile(wav_path) and idx not in self.invalid_utterance_ids:
                    corpus.new_file(wav_path, idx)

                    ll_raw = annotations.LabelList.create_single(
                        transcription_raw,
                        idx=audiomate.corpus.LL_WORD_TRANSCRIPT_RAW
                    )

                    ll_clean = annotations.LabelList.create_single(
                        transcription_clean,
                        idx=audiomate.corpus.LL_WORD_TRANSCRIPT
                    )

                    utterance = corpus.new_utterance(idx, idx, speaker_idx)
                    utterance.set_label_list(ll_raw)
                    utterance.set_label_list(ll_clean)

                    utt_ids.append(utterance.idx)

        return utt_ids
