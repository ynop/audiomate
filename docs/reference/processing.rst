audiomate.processing
====================

.. automodule:: audiomate.processing
.. currentmodule:: audiomate.processing

Processor
---------

.. autoclass:: Processor
   :members:

Pipeline
--------

.. automodule:: audiomate.processing.pipeline
.. currentmodule:: audiomate.processing.pipeline

.. autoclass:: audiomate.processing.pipeline.base.Step
   :members:

.. autoclass:: audiomate.processing.pipeline.base.Computation
   :members:

.. autoclass:: audiomate.processing.pipeline.base.Reduction
   :members:

Implementations
---------------

Some processing pipeline steps are already implemented.

.. _table-processing-step-implementations:

.. table:: Implementations of processing pipeline steps.

  ==============================  ===========
  Name                            Description
  ==============================  ===========
  MeanVarianceNorm                Normalizes features with given mean and variance.
  MelSpectrogram                  Exctracts MelSpectrogram features.
  MFCC                            Extracts MFCC features.
  ==============================  ===========

.. autoclass:: audiomate.processing.pipeline.MeanVarianceNorm
   :members:

.. autoclass:: audiomate.processing.pipeline.MelSpectrogram
   :members:

.. autoclass:: audiomate.processing.pipeline.MFCC
   :members:
