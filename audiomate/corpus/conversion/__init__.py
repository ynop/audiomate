"""
This module contains classes to convert the data of a corpus.
It is for example used to convert all audio data to wav files.

Audio File Conversion
---------------------
.. autoclass:: AudioFileConverter
   :members:
   :inherited-members:

.. autoclass:: WavAudioFileConverter
   :members:
   :inherited-members:

"""

from .base import AudioFileConverter  # noqa: F401

from .wav import WavAudioFileConverter  # noqa: F401
