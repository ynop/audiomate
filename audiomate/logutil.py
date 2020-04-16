""" Logging functionaly to use within audiomate. """

import logging
import time


PROGRESS_LOGGER_DELAY_SECONDS = 5 * 60
PROGRESS_LOGGER_LEVEL = logging.INFO


class MateLogger(logging.getLoggerClass()):
    """
    Extension of the default Logger,
    that adds functionality to log progress of an iterable.
    """

    def progress(self, iterable, total=None, description=None):
        """
        Log the the progress on the loop over the given ``iterable``.

        Arguments:
            iterable (Iterable): The iterable to iterate over.
            total (int): Total number of elements in the iterable.
                         If ``None``, tries to get it via ``len(iterable)``.

        Returns:
            generator: Just pipes through the iterable as a generator which is given.
        """

        if not self.isEnabledFor(PROGRESS_LOGGER_LEVEL):
            for x in iterable:
                yield x

        else:
            if total is None and iterable is not None:
                try:
                    total = len(iterable)
                except (TypeError, AttributeError):
                    total = None

            last_log = time.time()
            item_count = 0

            if description is not None:
                self.log(PROGRESS_LOGGER_LEVEL, '[Start] %s', description)

            for index, x in enumerate(iterable):
                item_count += 1
                current_time = time.time()

                if current_time - last_log > PROGRESS_LOGGER_DELAY_SECONDS:
                    self.log(
                        PROGRESS_LOGGER_LEVEL,
                        '[%d / %d] %s', index + 1, total or '?', description or ''
                    )
                    last_log = current_time

                yield x

            self.log(
                PROGRESS_LOGGER_LEVEL,
                '[Done %d] %s', total or item_count, description or ''
            )


# Update Logger class with our customized one
logging.setLoggerClass(MateLogger)


def getLogger():
    """ Returns the library based logger.  """
    return logging.getLogger('audiomate')
