"""
Wrapping opening function of audioread library.
This is used to cache the available backends.
If backend evaluation is done on every call it is very inefficient.
"""

import audioread

available_backends = audioread.available_backends()


def audio_open(path):
    """
    Just calls ``audioread.audio_open``,
    but with backends cached in a global variable.
    Brings better performance, since available backends
    are evaluated only once.
    """
    return audioread.audio_open(path, backends=available_backends)
