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

Implementations
---------------

.. _table-format-support-of-readers-writers-by-format:

.. table:: Support for Reading and Writing by Format

  ==============================  =====  =======
  Format                          Read   Write
  ==============================  =====  =======
  Broadcast                       x
  Default                         x      x
  Kaldi                           x      x
  MUSAN                           x
  Google Speech Commands          x
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

