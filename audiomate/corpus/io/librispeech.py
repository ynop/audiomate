import os
import shutil

from audiomate.utils import download
from audiomate.utils import files

from . import base


SUBSETS = {
    'dev-clean': 'http://www.openslr.org/resources/12/dev-clean.tar.gz',
    'dev-other': 'http://www.openslr.org/resources/12/dev-other.tar.gz',
    'test-clean': 'http://www.openslr.org/resources/12/test-clean.tar.gz',
    'test-other': 'http://www.openslr.org/resources/12/test-other.tar.gz',
    'train-clean-100': 'http://www.openslr.org/resources/12/train-clean-100.tar.gz',
    'train-clean-360': 'http://www.openslr.org/resources/12/train-clean-360.tar.gz',
    'train-other-500': 'http://www.openslr.org/resources/12/train-other-500.tar.gz',
}


class LibriSpeechDownloader(base.CorpusDownloader):
    """
    Downloader for the LibriSpeech Dataset.

    Args:
        subsets (list): List of subsets to download.
                        If empty or ``None``, all subsets are downloaded.
        num_threads (int): Number of threads to use for download files.
        keep_archives (bool): If ``True``, keep downloaded archives after extraction.
    """

    def __init__(self, subsets=None, num_threads=1, keep_archives=False):
        self.subsets = subsets
        self.num_threads = num_threads

    @classmethod
    def type(cls):
        return 'librispeech'

    def _download(self, target_path):
        if self.subsets is None or len(self.subsets) == 0:
            to_download = SUBSETS.keys()
        else:
            to_download = self.subsets

        os.makedirs(target_path, exist_ok=True)

        for subset_name in to_download:
            if subset_name in SUBSETS.keys():
                tmp_file = os.path.join(target_path, '{}.tar.gz'.format(subset_name))
                download.download_file(
                    SUBSETS[subset_name],
                    tmp_file,
                    show_progress=True,
                    num_threads=self.num_threads
                )
                download.extract_tar(tmp_file, target_path)
                extract_sub_path = os.path.join(target_path, 'LibriSpeech')

                for item in os.listdir(extract_sub_path):
                    item_path = os.path.join(extract_sub_path, item)
                    item_target_path = os.path.join(target_path, item)
                    shutil.move(item_path, item_target_path)

                shutil.rmtree('extract_sub_path', ignore_errors=True)

                os.remove(tmp_file)
