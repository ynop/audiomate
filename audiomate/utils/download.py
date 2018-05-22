"""
This module contains any functionality for downloading and extracting data from any remotes.
"""

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
