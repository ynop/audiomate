audiomate.corpus.preprocessing
==============================

.. automodule:: audiomate.corpus.preprocessing
.. currentmodule:: audiomate.corpus.preprocessing

Processors
----------

.. autoclass:: Processor
   :members:

.. autoclass:: OfflineProcessor
   :members:

Pipeline
--------

.. automodule:: audiomate.corpus.preprocessing.pipeline
.. currentmodule:: audiomate.corpus.preprocessing.pipeline

.. autoclass:: audiomate.corpus.preprocessing.pipeline.base.Step
   :members:

.. autoclass:: audiomate.corpus.preprocessing.pipeline.base.Computation
   :members:

.. autoclass:: audiomate.corpus.preprocessing.pipeline.base.Reduction
   :members:

As for the processor there are different subclasses for either offline or online pipelines.

.. autoclass:: audiomate.corpus.preprocessing.pipeline.offline.OfflineComputation
   :members:

.. autoclass:: audiomate.corpus.preprocessing.pipeline.offline.OfflineReduction
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

.. autoclass:: audiomate.corpus.preprocessing.pipeline.offline.MeanVarianceNorm
   :members:

.. autoclass:: audiomate.corpus.preprocessing.pipeline.offline.MelSpectrogram
   :members:

.. autoclass:: audiomate.corpus.preprocessing.pipeline.offline.MFCC
   :members:
