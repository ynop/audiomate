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


  ==============================  ========  =====  =======
  Format                          Download  Read   Write
  ==============================  ========  =====  =======
  Acoustic Event Dataset                    x
  Broadcast                                 x
  Default                                   x      x
  ESC-50                          x         x
  Free-Spoken-Digit-Dataset       x         x
  Folder                                    x
  Google Speech Commands                    x
  GTZAN                                     x
  Kaldi                                     x      x
  Mozilla DeepSpeech                               x
  MUSAN                                     x
  Tatoeba                         x         x
  TIMIT                                     x
  TUDA German Distant Speech                x
  Urbansound8k                              x
  VoxForge                        x         x
  ==============================  ========  =====  =======

Acoustic Event Dataset
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AEDReader
   :members:

Broadcast
^^^^^^^^^
.. autoclass:: BroadcastReader
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

Google Speech Commands
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SpeechCommandsReader
   :members:

GTZAN
^^^^^
.. autoclass:: GtzanReader
   :members:

Kaldi
^^^^^
.. autoclass:: KaldiReader
   :members:

.. autoclass:: KaldiWriter
   :members:

Mozilla DeepSpeech
^^^^^^^^^^^^^^^^^^
.. autoclass:: MozillaDeepSpeechWriter
   :members:

MUSAN
^^^^^
.. autoclass:: MusanReader
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
