""" Logging functionality to use within audiomate. """

import logging

import tqdm


# ==================================================================================================


class MateLogger(logging.Logger):
    """ Extension of the default Logger, which adds functionality to log progress of an iterable """

    def __init__(self, name, level=logging.INFO):
        logging.Logger.__init__(self, name, level)

    @staticmethod
    def progress(iterable, total=None, description=None):
        """ Interface for progress logging, that the used library can be exchanged easily """

        for x in tqdm.tqdm(iterable, desc=description, total=total):
            yield x


# ==================================================================================================

# Update Logger class with our customized one
logging.basicConfig()
logging.setLoggerClass(MateLogger)


# ==================================================================================================


def getLogger():
    """ Returns the library based logger """
    return logging.getLogger("audiomate")
