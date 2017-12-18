import abc


class CorpusReader(metaclass=abc.ABCMeta):
    """
    Abstract class for reading a corpus.

    To implement a reader for a custom format, programmers are expected to subclass this class and to implement all
    abstract methods. The documentation of each abstract methods details the requirements that have to be met by an
    implementation.
    """

    def load(self, path):
        """
        Load and return the corpus from the given path.

        Args:
            path (str): Path to the data set to load.

        Returns:
            Corpus: The loaded corpus

        Raises:
            IOError: When the data set is invalid, for example because required files (annotations, …) are missing.
        """
        # Check for missing files
        missing_files = self._check_for_missing_files(path)

        if missing_files is not None:
            raise IOError('Invalid data set of type {}: files {} not found at {}'.format(
                self.type(), ' '.join(missing_files), path))

        return self._load(path)

    @classmethod
    @abc.abstractmethod
    def type(cls):
        """
        Returns a string that uniquely identifies the reader. This is usually the name of the corpus, for example
        `musan` or `timit`. Users can use this string to obtain an instance of the desired reader through
        :meth:`~pingu.corpus.io.create_reader_of_type` or get a list of all built-in readers with
        :meth:`~pingu.corpus.io.available_readers`.

        Returns:
            str: Name of the reader
        """
        return 'not_implemented'

    @abc.abstractmethod
    def _load(self, path):
        """
        Performs the actual reading of the corpus.

        Implementations do not have to call :meth:`~pingu.corpus.io.CorpusReader._check_for_missing_files` themselves.
        This is automatically done by :meth:`~pingu.corpus.io.CorpusReader.load`.

        Args:
            path (str): Path to a directory where the data set resides.

        Returns:
            Corpus: The loaded corpus
        """
        pass

    @abc.abstractmethod
    def _check_for_missing_files(self, path):
        """
        Return a list of necessary files for the current type of data set that are missing in the given folder.
        None if path seems valid.
        """
        return None


class CorpusWriter(metaclass=abc.ABCMeta):
    """
    Abstract class for writing a corpus.

    To implement a writer for a custom format, programmers are expected to subclass this class and to implement all
    abstract methods. The documentation of each abstract methods details the requirements that have to be met by an
    implementation.
    """

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
        """
        Returns a string that uniquely identifies the reader. This is usually the name of the corpus, for example
        `musan` or `timit`. Users can use this string to obtain an instance of the desired reader through
        :meth:`~pingu.corpus.io.create_writer_of_type` or get a list of all built-in readers with
        :meth:`~pingu.corpus.io.available_writers`.

        Returns:
            str: Name of the writer
        """
        return 'not_implemented'

    @abc.abstractmethod
    def _save(self, corpus, path):
        """
        Writes the corpus to disk to the given path.

        Args:
            corpus (Corpus): Corpus to write to disk
            path (str): Path of the target directory
        """
        pass
