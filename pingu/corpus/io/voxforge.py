import os
import re
import tarfile
import shutil

import requests

from . import base

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
    """

    def __init__(self, lang='de', url=None):
        self.url = url

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
        downloaded = VoxforgeDownloader.download_files(available, temp_folder)
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

    @staticmethod
    def download_files(file_urls, target_path):
        """ Download all files and store to the given path. """
        os.makedirs(target_path, exist_ok=True)
        downloaded_files = []

        for file_url in file_urls:
            req = requests.get(file_url)

            if req.status_code != 200:
                raise base.FailedDownloadException('Failed to download file {} (status {})!'.format(req.status_code,
                                                                                                    file_url))

            file_name = os.path.basename(file_url)
            target_file_path = os.path.join(target_path, file_name)

            with open(target_file_path, 'wb') as f:
                f.write(req.content)

            downloaded_files.append(target_file_path)

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
