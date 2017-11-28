Welcome to pingu's documentation!
=================================

Pingu is a library for easy access to audio datasets. It provides the datastructures for accessing/loading different datasets in a generic way.
This should ease the use of audio datasets for example for machine learning tasks.

Example for loading a corpus and using the FramedSignalGrabber to retrieve the audio signal in frames::

    >>> ds = Corpus.load('/path/to/the/dataset', loader='default')
    >>> grabber = FramedSignalGrabber(ds, label_list_idx='music', frame_length=400, hop_size=160)
    >>>
    >>> # Every frame contains the actual signal and a vector defining the active labels
    >>> for frame in grabber:
    >>>     print(frame)
    (array([-0.00317392,  0.00866726,  0.01651051, ..., -0.01336711,
       -0.01263466, -0.01232948], dtype=float32), array([ 0.,  0.,  1.,  0.], dtype=float32))
    ...

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Notes

    notes/installation

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Documentation

    documentation/formats

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Package Reference

    reference/corpus
    reference/assets
    reference/io
    reference/grabber
    reference/subview
    reference/formats
    reference/utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`