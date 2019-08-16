"""
Wrapping opening function of audioread library.
This is used to cache the available backends.
If backend evaluation is done on every call it is very inefficient.
"""

import audioread

available_backends = audioread.available_backends()


def audio_open(path):
    return audioread.audio_open(path, backends=available_backends)
