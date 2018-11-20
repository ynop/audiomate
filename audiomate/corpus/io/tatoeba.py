import os
import shutil

import audiomate
from audiomate import annotations
from audiomate import issuers
from audiomate.utils import download
from audiomate.utils import textfile
from . import base

SENTENCE_LIST_URL = 'http://downloads.tatoeba.org/exports/sentences.tar.bz2'
AUDIO_LIST_URL = 'http://downloads.tatoeba.org/exports/sentences_with_audio.tar.bz2'

META_FILENAME = 'meta.txt'


class TatoebaDownloader(base.CorpusDownloader):
    """
    Downloader for audio files from the tatoeba platform.

    .. seealso::

       `Tatoeba <https://tatoeba.org/>`_
          Website

    Args:
        include_languages (list): List of languages to download. If None all are downloaded.
        include_licenses (list): Sentences are downloaded only if their license is in this list.
                                 If None all licenses are included.
        load_empty_license (bool): Sentences with an empty license are not meant to be reused.
                                   If False these sentences are ignored.
    """

    def __init__(self, include_languages=None, include_licenses=None, include_empty_licence=False):
        self.include_languages = include_languages
        self.include_licenses = include_licenses
        self.include_empty_licence = include_empty_licence

    @classmethod
    def type(cls):
        return 'tatoeba'

    def _download(self, target_path):
        temp_path = os.path.join(target_path, 'temp')
        os.makedirs(temp_path, exist_ok=True)

        sentence_ark = os.path.join(temp_path, 'sentences.tar.bz2')
        sentence_list = os.path.join(temp_path, 'sentences.csv')
        audio_ark = os.path.join(temp_path, 'sentences_with_audio.tar.bz2')
        audio_list = os.path.join(temp_path, 'sentences_with_audio.csv')

        download.download_file(SENTENCE_LIST_URL, sentence_ark)
        download.download_file(AUDIO_LIST_URL, audio_ark)

        download.extract_tar(sentence_ark, temp_path)
        download.extract_tar(audio_ark, temp_path)

        audio_entries = self._load_audio_list(audio_list)
        sentences = self._load_sentence_list(sentence_list)

        valid_sentence_ids = set(audio_entries.keys()).intersection(set(sentences.keys()))

        # sent-id, username, lang, transcript
        all_records = [(k, audio_entries[k][0], sentences[k][0], sentences[k][1]) for k in valid_sentence_ids]

        meta_path = os.path.join(target_path, META_FILENAME)
        textfile.write_separated_lines(meta_path, all_records, separator='\t', sort_by_column=0)

        self._download_audio_files(all_records, target_path)

        shutil.rmtree(temp_path, ignore_errors=True)

    def _load_audio_list(self, path):
        """
        Load and filter the audio list.

        Args:
            path (str): Path to the audio list file.

        Returns:
            dict: Dictionary of filtered sentences (id : username, license, attribution-url)
        """

        result = {}

        for entry in textfile.read_separated_lines_generator(path, separator='\t', max_columns=4):
            for i in range(len(entry)):
                if entry[i] == '\\N':
                    entry[i] = None

            if len(entry) < 4:
                entry.extend([None] * (4 - len(entry)))

            if not self.include_empty_licence and entry[2] is None:
                continue

            if self.include_licenses is not None and entry[2] not in self.include_licenses:
                continue

            result[entry[0]] = entry[1:]

        return result

    def _load_sentence_list(self, path):
        """
        Load and filter the sentence list.

        Args:
            path (str): Path to the sentence list.

        Returns:
            dict: Dictionary of sentences (id : language, transcription)
        """

        result = {}

        for entry in textfile.read_separated_lines_generator(path, separator='\t', max_columns=3):
            if self.include_languages is None or entry[1] in self.include_languages:
                result[entry[0]] = entry[1:]

        return result

    def _download_audio_files(self, records, target_path):
        """
        Download all audio files based on the given records.
        """

        for record in records:
            audio_folder = os.path.join(target_path, 'audio', record[2])
            audio_file = os.path.join(audio_folder, '{}.mp3'.format(record[0]))
            os.makedirs(audio_folder, exist_ok=True)

            download_url = 'https://audio.tatoeba.org/sentences/{}/{}.mp3'.format(record[2], record[0])
            download.download_file(download_url, audio_file)


class TatoebaReader(base.CorpusReader):
    """
    Reader for audio data downloaded with the Tatoeba downloader.
    """

    @classmethod
    def type(cls):
        return 'tatoeba'

    def _check_for_missing_files(self, path):
        meta_file = os.path.join(path, META_FILENAME)

        if os.path.isfile(meta_file):
            return []
        else:
            return [meta_file]

    def _load(self, path):
        corpus = audiomate.Corpus(path=path)

        meta_file = os.path.join(path, META_FILENAME)
        records = textfile.read_separated_lines_generator(meta_file, separator='\t', max_columns=4)

        for record in records:
            idx = record[0]
            speaker_idx = record[1]
            language = record[2]
            transcript = record[3]

            file_path = os.path.join(path, 'audio', language, '{}.mp3'.format(idx))
            corpus.new_file(file_path, idx)

            if speaker_idx not in corpus.issuers.keys():
                issuer = issuers.Speaker(speaker_idx)
                corpus.import_issuers(issuer)

            utterance = corpus.new_utterance(idx, idx, speaker_idx)
            utterance.set_label_list(annotations.LabelList.create_single(transcript,
                                                                         idx=audiomate.corpus.LL_WORD_TRANSCRIPT_RAW))

        return corpus
