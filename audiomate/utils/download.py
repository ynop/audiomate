"""
This module contains any functionality for downloading and
extracting data from any remotes.
"""

import zipfile
import tarfile
import multiprocessing

import requests
from pget.down import Downloader

from audiomate import logutil

logger = logutil.getLogger()

PROGRESS_LOGGER_BYTE_DELAY = 1024 * 1024 * 100


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
        result = list(logger.progress(
            p.imap(_download_file, dl_items),
            total=len(dl_items),
            description='Download Files'
        ))

        return result


def _download_file(item):
    """ Helper function to pass (url, target) to ``download_file``. """
    return download_file(item[0], item[1])


def download_file(url, target_path, num_threads=1):
    """
    Download the file from the given `url` and store it at `target_path`.
    Return a tuple x (url, bool, str).
    x[0] contains the url.
    If download failed x[1] is ``False`` and x[2] contains some error message.
    If download was fine x[1] is ``True`` and x[2] contains the target-path.
    """

    if num_threads > 1:
        return download_file_parallel(
            url,
            target_path,
            num_threads=num_threads
        )

    r = requests.get(url, stream=True)

    if r.status_code != 200:
        return (url, False, 'Failed to download file {} (status {})!'.format(
            r.status_code,
            url
        ))

    file_size = int(requests.head(url).headers['Content-Length'])
    bytes_loaded = 0
    bytes_since_last_log = 0
    logger.info('Download file from "%s" with size: %d B', url, file_size)

    with open(target_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                bytes_loaded += 1024
                bytes_since_last_log += 1024

                if bytes_since_last_log >= PROGRESS_LOGGER_BYTE_DELAY:
                    logger.info('Download [%06.2f%%]', bytes_loaded / file_size * 100)
                    bytes_since_last_log = 0

    logger.info('Finished download')

    return (url, True, target_path)


def download_file_parallel(url, target_path, num_threads=1):
    """
    Download the file from the given `url` and store it at `target_path`.
    Return a tuple x (url, bool, str).
    x[0] contains the url.
    If download failed x[1] is ``False`` and x[2] contains some error message.
    If download was fine x[1] is ``True`` and x[2] contains the target-path.
    """

    downloader = Downloader(url, target_path, num_threads)
    downloader.start()

    # Wait until we know file size
    while downloader.total_length == 0:
        pass

    file_size = downloader.total_length
    logger.info('Download file from "%s" with size: %d B', url, file_size)

    bytes_at_last_log = 0

    def callback(x):
        nonlocal bytes_at_last_log

        if x.total_downloaded - bytes_at_last_log >= PROGRESS_LOGGER_BYTE_DELAY:
            logger.info('Download [%06.2f%%]', x.total_downloaded / file_size * 100)
            bytes_at_last_log = x.total_downloaded

    downloader.subscribe(callback, 10)
    downloader.wait_for_finish()

    logger.info('Finished download')

    return (url, True, target_path)


def extract_zip(zip_path, target_folder):
    """
    Extract the content of the zip-file at ``zip_path`` into
    ``target_folder``.
    """
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_folder)


def extract_tar(tar_path, target_folder):
    """
    Extract the content of the tar-file at ``tar_path`` into
    ``target_folder``.
    """
    with tarfile.open(tar_path, 'r') as archive:
        archive.extractall(target_folder)
