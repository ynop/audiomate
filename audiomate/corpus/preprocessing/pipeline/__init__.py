"""
This module contains classes for creating a preprocessing/feature-extraction pipeline.

There are different classes for offline and online processing, subclassing either
:py:class:`audiomate.corpus.preprocessing.OfflineProcessor` or
:py:class:`audiomate.corpus.preprocessing.OnlineProcessor`.

A pipeline consists of one of two types of steps. A computation step takes data from a previous step or the input and
processes it. A reduction step is used to merge outputs of multiple previous steps.
It takes outputs of all incoming steps and outputs a single data block.

The steps are managed as a directed graph,
which is built by passing the parent steps to the ``__init__`` method of a step.
Every step that is created has his own graph, but inherits all nodes and edges of the graphs of his parent steps.
"""
