"""
This module contains any functionality for downloading and extracting data from any remotes.
"""

import zipfile
import tarfile
import requests
import multiprocessing

from tqdm import tqdm


def download_files(url_to_target, num_threads=1):
    """
    Download multiple files.

    Args:
        url_to_target (dict): Dict with mapping from source-url
                              to target-path.
        num_threads (int): Number of threads to use.
    """
    dl_items = list(url_to_target.items())

    with multiprocessing.pool.ThreadPool(num_threads) as p:
        result = list(tqdm(
            p.imap(_download_file, dl_items),
            total=len(dl_items),
            desc='Download Files'
        ))

        return result


def _download_file(item):
    """ Helper function to pass (url, target) to ``download_file``. """
    return download_file(item[0], item[1])


def download_file(url, target_path):
    """
    Download the file from the given `url` and store it at `target_path`.
    Return a tuple x (url, bool, str).
    x[0] contains the url.
    If download failed x[1] is ``False`` and x[2] contains some error message.
    If download was fine x[1] is ``True`` and x[2] contains the target-path.
    """

    r = requests.get(url, stream=True)

    if r.status_code != 200:
        return (url, False, 'Failed to download file {} (status {})!'.format(
            r.status_code,
            url
        ))

    with open(target_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    return (url, True, target_path)


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
