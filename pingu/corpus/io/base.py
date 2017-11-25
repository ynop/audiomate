import abc


class CorpusLoader(metaclass=abc.ABCMeta):
    """
    A loader defines functionality to load and save a corpus from/to a given path.
    """

    def load(self, path):
        """
        Load and return the corpus from the given path.

        Args:
            path (str): Path to the dataset to load.

        Returns:
            Corpus: The loaded corpus.
        """

        # Check for missing files
        missing_files = self._check_for_missing_files(path)

        if missing_files is not None:
            raise IOError('Invalid dataset of type {}: files {} not found at {}'.format(self.type(), ' '.join(missing_files), path))

        return self._load(path)

    def save(self, corpus, path):
        """
        Save the dataset at the given path.

        Args:
            corpus (Corpus): The corpus to save.
            path (str): Path to save the corpus to.
        """
        self._save(corpus, path)

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
    def _save(self, corpus, path):
        """ The loader specific save function. """
        pass

    @abc.abstractmethod
    def _check_for_missing_files(self, path):
        """ Return a list of necessary files for the current type of dataset that are missing in the given folder. None if path seems valid. """
        return None
