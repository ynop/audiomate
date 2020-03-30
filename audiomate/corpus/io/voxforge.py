import os
import re
import tarfile
import shutil

import requests

import audiomate
from . import base
from audiomate import annotations
from audiomate import issuers
from audiomate.utils import textfile
from audiomate.utils import download
from audiomate import logutil

logger = logutil.getLogger()

DOWNLOAD_URL = {
    'de': 'http://www.repository.voxforge1.org/downloads/de/Trunk/Audio/Main/16kHz_16bit/',
    'en': 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/'
}


class VoxforgeDownloader(base.CorpusDownloader):
    """
    Downloader for audio files from http://www.voxforge.org/.
    All .tgz files that are linked from the given url are downloaded and extracted.

    Args:
        lang (str): If no URL is given the predefined URL's for the given language is used, if one is defined.
        url (str): The url to check for available .tgz files.
        num_workers (int): Number of processes/threads to use for download.
    """

    def __init__(self, lang='de', url=None, num_workers=4):
        self.url = url
        self.num_workers = num_workers

        if url is None:
            if lang in DOWNLOAD_URL.keys():
                self.url = DOWNLOAD_URL[lang]
            else:
                raise ValueError('There is no voxforge URL present for language {}!'.format(lang))

    @classmethod
    def type(cls):
        return 'voxforge'

    def _download(self, target_path):
        temp_folder = os.path.join(target_path, 'download')
        os.makedirs(temp_folder, exist_ok=True)

        available = VoxforgeDownloader.available_files(self.url)
        logger.info('Found %d available archives to download', len(available))

        downloaded = self.download_files(available, temp_folder)

        logger.info('Extract %d files', len(downloaded))
        VoxforgeDownloader.extract_files(downloaded, target_path)

        shutil.rmtree(temp_folder)

    @staticmethod
    def available_files(url):
        """ Extract and return urls for all available .tgz files. """
        req = requests.get(url)

        if req.status_code != 200:
            raise base.FailedDownloadException('Failed to download data (status {}) from {}!'.format(req.status_code,
                                                                                                     url))

        page_content = req.text
        link_pattern = re.compile(r'<a href="(.*?)">(.*?)</a>')
        available_files = []

        for match in link_pattern.findall(page_content):
            if match[0].endswith('.tgz'):
                available_files.append(os.path.join(url, match[0]))

        return available_files

    def download_files(self, file_urls, target_path):
        """ Download all files and store to the given path. """
        os.makedirs(target_path, exist_ok=True)

        url_to_target = {}

        for file_url in file_urls:
            file_name = os.path.basename(file_url)
            target_file_path = os.path.join(target_path, file_name)
            url_to_target[file_url] = target_file_path

        dl_result = download.download_files(url_to_target, num_threads=self.num_workers)

        downloaded_files = []
        for url, status, path_or_msg in dl_result:
            if status:
                downloaded_files.append(path_or_msg)
            else:
                logger.info('Download failed for url %s', url)

        return downloaded_files

    @staticmethod
    def extract_files(file_paths, target_path):
        """ Unpack all files to the given path. """
        os.makedirs(target_path, exist_ok=True)
        extracted = []

        for file_path in file_paths:
            with tarfile.open(file_path, 'r') as archive:
                archive.extractall(target_path)

            file_name = os.path.splitext(os.path.basename(file_path))[0]
            extracted.append(os.path.join(target_path, file_name))

        return extracted


class VoxforgeReader(base.CorpusReader):
    """
    Reader for collections of voxforge audio data. The reader expects extracted .tgz files in the given folder.

    .. seealso::

       `<http://www.voxforge.org/>`_
          Download page
    """

    @classmethod
    def type(cls):
        return 'voxforge'

    def _check_for_missing_files(self, path):
        return []

    def _load(self, path):
        corpus = audiomate.Corpus(path=path)

        for dir_path in sorted(VoxforgeReader.data_folders(path)):
            item = os.path.basename(dir_path)
            etc_folder = os.path.join(dir_path, 'etc')
            wav_folder = os.path.join(dir_path, 'wav')
            readme_path = os.path.join(etc_folder, 'README')

            # LOAD ISSUER
            issuer = VoxforgeReader.parse_speaker_info(readme_path)

            if issuer.idx is None or issuer.idx == 'anonymous':
                issuer.idx = item

            # LOAD TRANSCRIPTIONS
            prompts, prompts_orig = VoxforgeReader.parse_prompts(etc_folder)

            # LOAD FILES/UTTS
            for file_name in os.listdir(wav_folder):
                wav_path = os.path.join(wav_folder, file_name)
                basename, ext = os.path.splitext(file_name)
                idx = '{}-{}'.format(item, basename)

                is_valid_wav = os.path.isfile(wav_path) and ext == '.wav' \
                    and idx not in self.invalid_utterance_ids
                has_transcription = basename in prompts.keys()

                if is_valid_wav and has_transcription:
                    if issuer.idx not in corpus.issuers.keys():
                        corpus.import_issuers([issuer])

                    corpus.new_file(wav_path, idx)
                    utt = corpus.new_utterance(idx, idx, issuer.idx)
                    utt.set_label_list(annotations.LabelList.create_single(prompts[basename],
                                                                           idx=audiomate.corpus.LL_WORD_TRANSCRIPT))

                    if basename in prompts_orig.keys():
                        raw = annotations.LabelList.create_single(prompts_orig[basename],
                                                                  idx=audiomate.corpus.LL_WORD_TRANSCRIPT_RAW)
                        utt.set_label_list(raw)

        return corpus

    @staticmethod
    def data_folders(path):
        """ Generator which yields a list of valid data directories (corresponds to the content of one .tgz). """
        for item in os.listdir(path):
            dir_path = os.path.join(path, item)
            wav_folder = os.path.join(dir_path, 'wav')

            if os.path.isdir(dir_path) and os.path.isdir(wav_folder):
                yield dir_path

    @staticmethod
    def parse_speaker_info(readme_path):
        """ Parse speaker info and return tuple (idx, gender). """
        idx = None
        gender = issuers.Gender.UNKNOWN
        age_group = issuers.AgeGroup.UNKNOWN
        native_lang = None

        with open(readme_path, 'r', errors='ignore') as f:
            for raw_line in f:
                line = raw_line.strip()

                if line is not None and line != '':
                    line = line.rstrip(';.')
                    parts = line.split(':', maxsplit=1)

                    if len(parts) > 1:
                        key = parts[0].strip().lower()
                        value = parts[1].strip()

                        if key == 'user name':
                            idx = value

                        value = value.lower()

                        if key == 'gender':
                            if value in ['männlich', 'male', 'mnnlich']:
                                gender = issuers.Gender.MALE
                            elif value in ['weiblich', 'female', '[female]']:
                                gender = issuers.Gender.FEMALE

                        if key == 'age range':
                            if value in ['erwachsener', 'adult', '[adult]', '[erwachsener]']:
                                age_group = issuers.AgeGroup.ADULT
                            elif value in ['senior', '[senior']:
                                age_group = issuers.AgeGroup.SENIOR
                            elif value in ['youth', 'jugendlicher', '[youth]', '[jugendlicher]']:
                                age_group = issuers.AgeGroup.YOUTH
                            elif value in ['kind', 'child']:
                                age_group = issuers.AgeGroup.CHILD

                        if key == 'language':
                            if value in ['de', 'ger', 'deu', '[de]']:
                                native_lang = 'deu'
                            elif value in ['en', 'eng', '[en]']:
                                native_lang = 'eng'

        return issuers.Speaker(idx, gender=gender, age_group=age_group, native_language=native_lang)

    @staticmethod
    def parse_prompts(etc_folder):
        """ Read prompts and prompts-orignal and return as dictionary (id as key). """
        prompts_path = os.path.join(etc_folder, 'PROMPTS')
        prompts_orig_path = os.path.join(etc_folder, 'prompts-original')

        prompts = textfile.read_key_value_lines(prompts_path, separator=' ')
        prompts_orig = textfile.read_key_value_lines(prompts_orig_path, separator=' ')

        prompts_key_fixed = {}

        for k, v in prompts.items():
            parts = k.split('/')
            key = k

            if len(parts) > 1:
                key = parts[-1]

            prompts_key_fixed[key] = v

        prompts = prompts_key_fixed

        return prompts, prompts_orig
