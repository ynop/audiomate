"""
This module contains functionality for creating any kind of subsets from a corpus.
A subset of a corpus is represented with a :py:class:`Subview`.
The data contained in a subview is defined by one or more :py:class:`FilterCriterion`.

For creating subviews there are additional classes.
:py:class:`Splitter` can be used to divide a corpus into subsets according to given proportions.
:py:class:`SubsetGenerator` can be used to create subset with given settings.


Subview
-------
.. autoclass:: Subview
   :members:
   :inherited-members:


Filter
------
.. autoclass:: FilterCriterion
   :members:
   :inherited-members:


MatchingUtteranceIdxFilter
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: MatchingUtteranceIdxFilter


MatchingLabelFilter
^^^^^^^^^^^^^^^^^^^
.. autoclass:: MatchingLabelFilter


Splitter
--------
.. autoclass:: Splitter
   :members:
   :inherited-members:


SubsetGenerator
---------------

.. autoclass:: SubsetGenerator
   :members:
   :inherited-members:


Utils
-----

.. automodule:: audiomate.corpus.subset.utils
   :members:

"""

from .subview import FilterCriterion  # noqa: F401
from .subview import MatchingUtteranceIdxFilter  # noqa: F401
from .subview import MatchingLabelFilter  # noqa: F401

from .subview import Subview  # noqa: F401

from .splitting import Splitter  # noqa: F401
from .selection import SubsetGenerator  # noqa: F401
