"""
This module contains classes to read and write corpora from the filesystem in a wide range of formats. They can also be
used to convert between formats.
"""

from .base import CorpusDownloader, CorpusReader, CorpusWriter
from .downloader import ArchiveDownloader

from .broadcast import BroadcastReader  # noqa: F401
from .default import DefaultReader, DefaultWriter  # noqa: F401
from .gtzan import GtzanDownloader, GtzanReader  # noqa: F401
from .kaldi import KaldiReader, KaldiWriter  # noqa: F401
from .musan import MusanDownloader, MusanReader  # noqa: F401
from .speech_commands import SpeechCommandsReader  # noqa: F401
from .tuda import TudaReader  # noqa: F401
from .folder import FolderReader  # noqa: F401
from .esc import ESC50Downloader, ESC50Reader  # noqa: F401
from .mozilla_deepspeech import MozillaDeepSpeechWriter  # noqa: F401
from .voxforge import VoxforgeDownloader, VoxforgeReader  # noqa: F401
from .aed import AEDReader, AEDDownloader  # noqa: F401
from .urbansound import Urbansound8kReader  # noqa: F401
from .timit import TimitReader  # noqa: F401
from .swc import SWCReader  # noqa: F401
from .free_spoken_digits import FreeSpokenDigitDownloader, FreeSpokenDigitReader  # noqa: F401
from .tatoeba import TatoebaDownloader, TatoebaReader  # noqa: F401
from .common_voice import CommonVoiceReader  # noqa: F401
from .mailabs import MailabsDownloader, MailabsReader  # noqa: F401
from .rouen import RouenDownloader, RouenReader  # noqa: F401
from .audio_mnist import AudioMNISTDownloader, AudioMNISTReader  # noqa: F401

__downloaders = {}
for cls in CorpusDownloader.__subclasses__():
    if cls != ArchiveDownloader:
        __downloaders[cls.type()] = cls

__readers = {}
for cls in CorpusReader.__subclasses__():
    __readers[cls.type()] = cls

__writers = {}
for cls in CorpusWriter.__subclasses__():
    __writers[cls.type()] = cls


class UnknownDownloaderException(Exception):
    pass


class UnknownReaderException(Exception):
    pass


class UnknownWriterException(Exception):
    pass


def available_downloaders():
    """
    Get a mapping of all available downloaders.

    Returns:
        dict: A dictionary with downloader classes with the name of these downloaders as key.

    Example::

        >>> available_downloaders()
        {
            "voxforge" : audiomate.corpus.io.VoxforgeDownloader
        }
    """
    return __downloaders


def available_readers():
    """
    Get a mapping of all available readers.

    Returns:
        dict: A dictionary with reader classes with the name of these readers as key.

    Example::

        >>> available_readers()
        {
            "default" : audiomate.corpus.io.DefaultReader,
            "kaldi" : audiomate.corpus.io.KaldiReader
        }
    """
    return __readers


def available_writers():
    """
    Get a mapping of all available writers.

    Returns:
        dict: A dictionary with writer classes with the name of these writers as key.

    Example::

        >>> available_writers()
        {
            "default" : audiomate.corpus.io.DefaultWriter,
            "kaldi" : audiomate.corpus.io.KaldiWriter
        }
    """
    return __writers


def create_downloader_of_type(type_name):
    """
        Create an instance of the downloader with the given name.

        Args:
            type_name: The name of a downloader.

        Returns:
            An instance of the downloader with the given type.
    """
    downloaders = available_downloaders()

    if type_name not in downloaders.keys():
        raise UnknownDownloaderException('Unknown downloader: %s' % (type_name,))

    return downloaders[type_name]()


def create_reader_of_type(type_name):
    """
        Create an instance of the reader with the given name.

        Args:
            type_name: The name of a reader.

        Returns:
            An instance of the reader with the given type.
    """
    readers = available_readers()

    if type_name not in readers.keys():
        raise UnknownReaderException('Unknown reader: %s' % (type_name,))

    return readers[type_name]()


def create_writer_of_type(type_name):
    """
        Create an instance of the writer with the given name.

        Args:
            type_name: The name of a writer.

        Returns:
            An instance of the writer with the given type.
    """
    writers = available_writers()

    if type_name not in writers.keys():
        raise UnknownWriterException('Unknown writer: %s' % (type_name,))

    return writers[type_name]()
