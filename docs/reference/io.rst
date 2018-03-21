pingu.corpus.io
===============

.. automodule:: pingu.corpus.io
    :members:
.. currentmodule:: pingu.corpus.io

Base Classes
------------

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

  ==============================  =====  =======
  Format                          Read   Write
  ==============================  =====  =======
  Broadcast                       x
  Default                         x      x
  Folder                          x
  GTZAN                           x
  Kaldi                           x      x
  MUSAN                           x
  Google Speech Commands          x
  TUDA German Distant Speech      x
  ESC-50                          x
  Mozilla DeepSpeech                     x
  ==============================  =====  =======


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

Folder
^^^^^^
.. autoclass:: FolderReader
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

MUSAN
^^^^^
.. autoclass:: MusanReader
   :members:

Google Speech Commands
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SpeechCommandsReader
   :members:

TUDA German Distant Speech
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: TudaReader
   :members:

ESC-50
^^^^^^
.. autoclass:: ESC50Reader
   :members:

Mozilla DeepSpeech
^^^^^^^^^^^^^^^^^^
.. autoclass:: MozillaDeepSpeechWriter
   :members:


