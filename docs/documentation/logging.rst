.. _logging:

Logging
=======

Logging in audiomate is done using the standard Python logging facilities.

Enable Logging
--------------

By default, only messages of severity ``Warning`` or higher are printed to ``sys.stderr``.
Audiomate provides detailed information about progress of long-running tasks with messages of severity ``Info``.
To enable logging of messages of lower severity, configure Python's logging system as follows:

.. code-block:: python

    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)-15s  %(name)s  %(message)s'
    )

For further information check the python `logging documentation <https://docs.python.org/3/howto/logging.html>`_.

Create log messages in audiomate
--------------------------------

Logging in audiomate is done with a single logger.
The logger is available in :mod:`audiomate.logutil`.

.. code-block:: python

    from audiomate import logutil

    logger = logutil.getLogger()

    def some_functionality():
        logger.debug('message')

Since audiomate has a lot of long-running tasks,
a special function for logging the progress of a loop can be used.
It basically is a wrapper around an iterable to check and log the progress.
In order to keep the logs as small as possible,
progress is logged in steps of 5 minutes.


.. code-block:: python

    from audiomate import logutil

    logger = logutil.getLogger()

    for utterance in logger.progress(
            corpus.utterances.values(),
            total=corpus.num_utterances,
            description='Process utterances'):

        # Do something with the utterance,
        # that takes up some time.
