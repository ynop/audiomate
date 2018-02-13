pingu.corpus.preprocessing
==========================

.. automodule:: pingu.corpus.preprocessing
.. currentmodule:: pingu.corpus.preprocessing

Processors
----------

.. autoclass:: Processor
   :members:

.. autoclass:: OfflineProcessor
   :members:

Pipeline
--------

.. automodule:: pingu.corpus.preprocessing.pipeline
.. currentmodule:: pingu.corpus.preprocessing.pipeline

.. autoclass:: pingu.corpus.preprocessing.pipeline.base.Step
   :members:

.. autoclass:: pingu.corpus.preprocessing.pipeline.base.Computation
   :members:

.. autoclass:: pingu.corpus.preprocessing.pipeline.base.Reduction
   :members:

As for the processor there are different subclasses for either offline or online pipelines.

.. autoclass:: pingu.corpus.preprocessing.pipeline.offline.OfflineComputation
   :members:

.. autoclass:: pingu.corpus.preprocessing.pipeline.offline.OfflineReduction
   :members:

Implementations
---------------

Some preprocessing steps are already implemented.

Offline
^^^^^^^

.. _table-preprocessing-step-implementations-offline:

.. table:: Implementations of offline preprocessing steps.

  ==============================  ===========
  Name                            Description
  ==============================  ===========
  MeanVarianceNorm                Normalizes features with given mean and variance.
  MelSpectrogram                  Exctracts MelSpectrogram features.
  MFCC                            Extracts MFCC features.
  ==============================  ===========

.. autoclass:: pingu.corpus.preprocessing.pipeline.offline.MeanVarianceNorm
   :members:

.. autoclass:: pingu.corpus.preprocessing.pipeline.offline.MelSpectrogram
   :members:

.. autoclass:: pingu.corpus.preprocessing.pipeline.offline.MFCC
   :members:
