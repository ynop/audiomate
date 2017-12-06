"""
The io module contains the loaders which provide functionality to load corpora from the filesystem.

All loader implementations base on :py:class:`pingu.corpus.io.CorpusLoader`.
"""

from .base import CorpusLoader
from .broadcast import BroadcastLoader  # noqa: F401
from .default import DefaultLoader  # noqa: F401
from .kaldi import KaldiLoader  # noqa: F401
from .musan import MusanLoader  # noqa: F401

__loaders = {}
for cls in CorpusLoader.__subclasses__():
    __loaders[cls.type()] = cls


class UnknownLoaderException(Exception):
    pass


def available_loaders():
    """
    Get a mapping of all available loaders.

    Returns:
        dict: A dictionary with loader classes with the name of these loaders as key.

    Example::

        >>> available_loaders()
        {
            "default" : pingu.corpus.io.DefaultLoader,
            "kaldi" : pingu.corpus.io.KaldiLoader
        }
    """
    return __loaders


def create_loader_of_type(type_name):
    """
        Create an instance of the loader with the given name.

        Args:
            type_name: The name of a loader.

        Returns:
            An instance of the loader with the given type.
    """
    loaders = available_loaders()

    if type_name not in loaders.keys():
        raise UnknownLoaderException('Unknown loader: %s' % (type_name,))

    return loaders[type_name]()
