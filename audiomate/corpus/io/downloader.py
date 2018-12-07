import abc
import enum
import os
import tarfile
import zipfile

from audiomate.utils import download
from audiomate.utils import files
from . import base


class ArkType(enum.Enum):
    """
    Enum defining different types of archives.
    """
    ZIP = 1
    TAR = 2
    AUTO = 3


class ArchiveDownloader(base.CorpusDownloader, abc.ABC):
    """
    Convenience base class for a downloader of a corpus,
    that consists of a single archive.

    Args:
        url (str): URL, from where to download the archive.
        ark_type (ArkType): The type of the archive.
                            If ``AUTO`` it tries to find the type
                            automatically.
        move_files_up (bool): If ``True`` moves all files/folders
                              from subfolders to the root-folder.
    """

    def __init__(self, url, ark_type=ArkType.AUTO, move_files_up=False):
        self.url = url
        self.ark_type = ark_type
        self.move_files_up = move_files_up

    def _download(self, target_path):
        os.makedirs(target_path, exist_ok=True)
        tmp_file = os.path.join(target_path, 'tmp_ark')

        download.download_file(self.url, tmp_file)
        self._extract_file(tmp_file, target_path)

        if self.move_files_up:
            files.move_all_files_from_subfolders_to_top(
                target_path,
                delete_subfolders=True
            )

        os.remove(tmp_file)

    def _extract_file(self,  file_path, target_folder):
        ark_type = self.ark_type

        if self.ark_type == ArkType.AUTO:
            if tarfile.is_tarfile(file_path):
                ark_type = ArkType.TAR
            elif zipfile.is_zipfile(file_path):
                ark_type = ArkType.ZIP

        if ark_type == ArkType.TAR:
            download.extract_tar(file_path, target_folder)
        elif ark_type == ArkType.ZIP:
            download.extract_zip(file_path, target_folder)
        else:
            raise ValueError(
                'Unrecognized archive type (Only zip/tar supported)!'
            )
