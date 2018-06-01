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

.. autoclass:: audiomate.processing.pipeline.Chunk
   :members:

.. autoclass:: audiomate.processing.pipeline.Step
   :members:

.. autoclass:: audiomate.processing.pipeline.Computation
   :members:

.. autoclass:: audiomate.processing.pipeline.Reduction
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
  PowerToDb                       Convert power spectrum to Db.
  Delta                           Compute delta features.
  Stack                           Reduce multiple features into one by stacking them on top of each other.
  ==============================  ===========

.. autoclass:: audiomate.processing.pipeline.MeanVarianceNorm
   :members:

.. autoclass:: audiomate.processing.pipeline.MelSpectrogram
   :members:

.. autoclass:: audiomate.processing.pipeline.MFCC
   :members:

.. autoclass:: audiomate.processing.pipeline.PowerToDb
   :members:

.. autoclass:: audiomate.processing.pipeline.Delta
   :members:

.. autoclass:: audiomate.processing.pipeline.Stack
   :members:
