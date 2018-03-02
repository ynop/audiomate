"""
This module contains classes to read and write corpora from the filesystem in a wide range of formats. They can also be
used to convert between formats.
"""

from .base import CorpusReader, CorpusWriter
from .broadcast import BroadcastReader  # noqa: F401
from .default import DefaultReader, DefaultWriter  # noqa: F401
from .gtzan import GtzanReader  # noqa: F401
from .kaldi import KaldiReader, KaldiWriter  # noqa: F401
from .musan import MusanReader  # noqa: F401
from .speech_commands import SpeechCommandsReader  # noqa: F401
from .folder import FolderReader  # noqa: F401

__readers = {}
for cls in CorpusReader.__subclasses__():
    __readers[cls.type()] = cls

__writers = {}
for cls in CorpusWriter.__subclasses__():
    __writers[cls.type()] = cls


class UnknownReaderException(Exception):
    pass


class UnknownWriterException(Exception):
    pass


def available_readers():
    """
    Get a mapping of all available readers.

    Returns:
        dict: A dictionary with reader classes with the name of these readers as key.

    Example::

        >>> available_readers()
        {
            "default" : pingu.corpus.io.DefaultReader,
            "kaldi" : pingu.corpus.io.KaldiReader
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
            "default" : pingu.corpus.io.DefaultWriter,
            "kaldi" : pingu.corpus.io.KaldiWriter
        }
    """
    return __writers


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
