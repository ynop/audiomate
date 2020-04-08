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


  ================================  ========  =====  =======
  Format                            Download  Read   Write
  ================================  ========  =====  =======
  Acoustic Event Dataset            x         x
  AudioMNIST                        x         x
  Broadcast                                   x
  Common Voice                      x         x
  Default                                     x      x
  ESC-50                            x         x
  Free-Spoken-Digit-Dataset         x         x
  Folder                                      x
  Fluent Speech Commands Dataset              x
  Google Speech Commands                      x
  GTZAN                             x         x
  Kaldi                                       x      x
  LibriSpeech                       x         x
  Mozilla DeepSpeech                                 x
  MUSAN                             x         x
  M-AILABS Speech Dataset           x         x
  LITIS Rouen Audio scene dataset   x         x
  Spoken Wikipedia Corpora          x         x
  Tatoeba                           x         x
  TIMIT                                       x
  TUDA German Distant Speech        x         x
  Urbansound8k                                x
  VoxForge                          x         x
  Wav2Letter                                         x
  ================================  ========  =====  =======

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
