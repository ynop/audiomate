import abc


class CorpusLoader(metaclass=abc.ABCMeta):
    """
    A loader defines functionality to load and save a dataset from/to a given path.
    """

    def load(self, path):
        """ Load and return the dataset from the given path. """

        # Check for missing files
        missing_files = self._check_for_missing_files(path)

        if missing_files is not None:
            raise IOError('Invalid dataset of type {}: files {} not found at {}'.format(self.type(), ' '.join(missing_files), path))

        return self._load(path)

    def save(self, dataset, path):
        """ Save the dataset at the given path. """
        self._save(dataset, path)

    @classmethod
    @abc.abstractmethod
    def type(cls):
        """ Return the type of the loader (e.g. kaldi, TIMIT, ...). """
        return 'not_implemented'

    @abc.abstractmethod
    def _load(self, path):
        """ The loader specific load function. """
        pass

    @abc.abstractmethod
    def _save(self, dataset, path):
        """ The loader specific save function. """
        pass

    @abc.abstractmethod
    def _check_for_missing_files(self, path):
        """ Return a list of necessary files for the current type of dataset that are missing in the given folder. None if path seems valid. """
        return None
