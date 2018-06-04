"""
This module contains any functionality for downloading and extracting data from any remotes.
"""

import zipfile
import tarfile
import requests


def download_file(url, target_path):
    """
    Download the file from the given `url` and store it at `target_path`.
    """

    r = requests.get(url, stream=True)

    with open(target_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


def extract_zip(zip_path, target_folder):
    """
    Extract the content of the zip-file at `zip_path` into `target_folder`.
    """
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_folder)


def extract_tar(tar_path, target_folder):
    """
    Extract the content of the tar-file at `tar_path` into `target_folder`.
    """
    with tarfile.open(tar_path, 'r') as archive:
        archive.extractall(target_folder)
