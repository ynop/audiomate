audiomate.corpus.io
===================

.. automodule:: audiomate.corpus.io
    :members:
.. currentmodule:: audiomate.corpus.io

Base Classes
------------

.. autoclass:: CorpusDownloader
   :members:
   :inherited-members:
   :private-members:

.. autoclass:: ArchiveDownloader
   :members:
   :inherited-members:
   :private-members:

.. autoclass:: CorpusReader
   :members:
   :inherited-members:
   :private-members:

.. autoclass:: CorpusWriter
   :members:
   :inherited-members:
   :private-members:

.. _io_implementations:

Implementations
---------------

.. _table-format-support-of-readers-writers-by-format:

.. table:: Support for Reading and Writing by Format


  ================================  ========  =====  ======= ==================
  Format                            Download  Read   Write   Key (for reading and writing)
  ================================  ========  =====  ======= ==================
  Acoustic Event Dataset            x         x              aed
  AudioMNIST                        x         x              audio-mnist
  Broadcast                                   x              broadcast
  Common Voice                      x         x              common-voice
  Default                                     x      x       default
  ESC-50                            x         x              esc-50
  Free-Spoken-Digit-Dataset         x         x              free-spoken-digits
  Folder                                      x              folder
  Fluent Speech Commands Dataset              x              fluent-speech
  Google Speech Commands                      x              speech-commands
  GTZAN                             x         x              gtzan
  Kaldi                                       x      x       kaldi
  LibriSpeech                       x         x              librispeech
  Mozilla DeepSpeech                                 x       mozilla-deepspeech
  MUSAN                             x         x              musan
  M-AILABS Speech Dataset           x         x              mailabs
  NVIDIA Jasper                                      x       nvidia-jasper
  LITIS Rouen Audio scene dataset   x         x              rouen
  Spoken Wikipedia Corpora          x         x              swc
  Tatoeba                           x         x              tatoeba
  TIMIT                                       x              timit
  TUDA German Distant Speech        x         x              tuda
  Urbansound8k                                x              urbansound8k
  VoxForge                          x         x              voxforge
  Wav2Letter                                         x       wav2letter
  ================================  ========  =====  ======= ==================

Acoustic Event Dataset
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AEDDownloader
   :members:

.. autoclass:: AEDReader
   :members:

AudioMNIST
^^^^^^^^^^
.. autoclass:: AudioMNISTDownloader
   :members:

.. autoclass:: AudioMNISTReader
   :members:

Broadcast
^^^^^^^^^
.. autoclass:: BroadcastReader
   :members:

Common-Voice
^^^^^^^^^^^^
.. autoclass:: CommonVoiceDownloader
   :members:

.. autoclass:: CommonVoiceReader
   :members:

Default
^^^^^^^
.. autoclass:: DefaultReader
   :members:

.. autoclass:: DefaultWriter
   :members:

ESC-50
^^^^^^
.. autoclass:: ESC50Downloader
   :members:

.. autoclass:: ESC50Reader
   :members:

Folder
^^^^^^
.. autoclass:: FolderReader
   :members:

Free-Spoken-Digit-Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: FreeSpokenDigitDownloader
   :members:

.. autoclass:: FreeSpokenDigitReader
   :members:

Fluent Speech Commands Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: FluentSpeechReader
   :members:

Google Speech Commands
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SpeechCommandsReader
   :members:

GTZAN
^^^^^
.. autoclass:: GtzanDownloader
   :members:

.. autoclass:: GtzanReader
   :members:

Kaldi
^^^^^
.. autoclass:: KaldiReader
   :members:

.. autoclass:: KaldiWriter
   :members:

LibriSpeech
^^^^^^^^^^^
.. autoclass:: LibriSpeechDownloader
   :members:

.. autoclass:: LibriSpeechReader
   :members:

Mozilla DeepSpeech
^^^^^^^^^^^^^^^^^^
.. autoclass:: MozillaDeepSpeechWriter
   :members:

MUSAN
^^^^^
.. autoclass:: MusanDownloader
   :members:

.. autoclass:: MusanReader
   :members:

M-AILABS Speech Dataset
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: MailabsDownloader
   :members:

.. autoclass:: MailabsReader
   :members:

NVIDIA Jasper
^^^^^^^^^^^^^
.. autoclass:: NvidiaJasperWriter
   :members:

LITIS Rouen Audio scene dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: RouenDownloader
   :members:

.. autoclass:: RouenReader
   :members:

SWC - Spoken Wikipedia Corpora
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SWCDownloader
   :members:

.. autoclass:: SWCReader
   :members:

Tatoeba
^^^^^^^
.. autoclass:: TatoebaDownloader
   :members:

.. autoclass:: TatoebaReader
   :members:

TIMIT DARPA Acoustic-Phonetic Continuous Speech Corpus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: TimitReader
   :members:

TUDA German Distant Speech
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: TudaDownloader
   :members:

.. autoclass:: TudaReader
   :members:

Urbansound8k
^^^^^^^^^^^^
.. autoclass:: Urbansound8kReader
   :members:

VoxForge
^^^^^^^^

.. autoclass:: VoxforgeDownloader
   :members:

.. autoclass:: VoxforgeReader
   :members:

Wav2Letter
^^^^^^^^^^
.. autoclass:: Wav2LetterWriter
   :members:
