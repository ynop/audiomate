"""
The processing module provides tools for processing audio data in a batch-wise manner.
The idea is to setup a predefined tool that can process all the audio from a corpus.

The basic component is the :py:class:`audiomate.processing.Processor`. It provides the functionality
to reduce any input component like a corpus, feature-container, utterance, file to the abstraction of frames.
A concrete implementation then only has to provide the proper method to process these frames.

Often in audio processing the same components are used in combination with others.
For this purpose a pipeline can be built that processes the frames in multiple steps.
The :mod:`audiomate.processing.pipeline` provides the :py:class:`audiomate.processing.pipeline.Computation`
and :py:class:`audiomate.processing.pipeline.Reduction` classes.
These abstract classes can be extended to create processing components of a pipeline.
The different components are then be coupled to create custom pipelines.
"""

from .base import Processor  # noqa: F401
