import abc
import os
import shutil

from audiomate.utils import jsonfile


class FailedDownloadException(Exception):
    pass


class CorpusDownloader(metaclass=abc.ABCMeta):
    """
    Abstract class for downloading a corpus.

    To implement a downloader for a custom format, programmers are expected to subclass this class and to implement all
    abstract methods. The documentation of each abstract methods details the requirements that have to be met by an
    implementation.
    """

    def download(self, target_path, force_redownload=False):
        """
        Downloads the data of the corpus and saves it to the given path.
        The data has to be saved in a way, so that the corresponding ``CorpusReader`` can load the corpus.

        Args:
            target_path (str): The path to save the data to.
            force_redownload (bool, optional): If ``True``, overwrite the target path and redownload the corpus.

        Raises:
            IOError: When the corpus has already been downloaded to the target path.
                     Overridden if `force_redownload` is set to ``True``.
        """
        if os.path.exists(target_path) and len(os.listdir(target_path)) > 0:

            if not force_redownload:
                raise IOError('Corpus already downloaded at {}.'.format(target_path))
            shutil.rmtree(target_path)

        return self._download(target_path)

    @classmethod
    @abc.abstractmethod
    def type(cls):
        """
        Returns a string that uniquely identifies the downloader. This is usually the name of the corpus, for example
        `musan` or `timit`. Users can use this string to obtain an instance of the desired reader through
        :meth:`~audiomate.corpus.io.create_downloader_of_type` or get a list of all built-in downloaders with
        :meth:`~audiomate.corpus.io.available_downloaders`.

        Returns:
            str: Name of the downloader
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _download(self, target_path):
        """
        Performs the actual downloading of the corpus.

        Args:
            target_path (str): Path to a directory where the data should be saved to.
        """
        raise NotImplementedError()


class CorpusReader(metaclass=abc.ABCMeta):
    """
    Abstract class for reading a corpus.

    To implement a reader for a custom format, programmers are expected to subclass this class and to implement all
    abstract methods. The documentation of each abstract methods details the requirements that have to be met by an
    implementation.

    Args:
        include_invalid_items (bool): Some readers define a list of invalid utterances/files.
                                     (e.g. bad transcription, invalid audio, ...)
                                     If ``False``, those utterances are loaded anyway.
                                     If ``True``, those utterances are ignored.
    """

    def __init__(self, include_invalid_items=False):
        self.include_invalid_items = include_invalid_items
        if not self.include_invalid_items:
            self.invalid_utterance_ids = self._load_list_of_invalid_utterances()
        else:
            self.invalid_utterance_ids = []

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

        if len(missing_files) > 0:
            raise IOError('Invalid data set of type {}: files {} not found at {}'.format(
                self.type(), ' '.join(missing_files), path))

        return self._load(path)

    def _load_list_of_invalid_utterances(self):
        io_folder = os.path.dirname(__file__)
        invalid_utt_path = os.path.join(io_folder, 'data', self.type(), 'invalid_utterances.json')

        if os.path.isfile(invalid_utt_path):
            return jsonfile.read_json_file(invalid_utt_path)
        else:
            return []

    @classmethod
    @abc.abstractmethod
    def type(cls):
        """
        Returns a string that uniquely identifies the reader. This is usually the name of the corpus, for example
        `musan` or `timit`. Users can use this string to obtain an instance of the desired reader through
        :meth:`~audiomate.corpus.io.create_reader_of_type` or get a list of all built-in readers with
        :meth:`~audiomate.corpus.io.available_readers`.

        Returns:
            str: Name of the reader
        """
        return 'not_implemented'

    @abc.abstractmethod
    def _load(self, path):
        """
        Performs the actual reading of the corpus.

        Implementations do not have to call :meth:`~audiomate.corpus.io.CorpusReader._check_for_missing_files`
        themselves. This is automatically done by :meth:`~audiomate.corpus.io.CorpusReader.load`.

        Args:
            path (str): Path to a directory where the data set resides.

        Returns:
            Corpus: The loaded corpus
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _check_for_missing_files(self, path):
        """
        Tests whether all required files (like annotations) to read the corpus successfully are present. If files are
        missing, a list with the paths of the missing files is returned. All paths are relative to `path`. If no files
        are missing, an empty list is returned.

        Args:
            path (str): Path to the root directory of the data set

        Returns:
            list: Paths of all the missing files, relative to the path of the root directory of the data set.
        """
        return []


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
        :meth:`~audiomate.corpus.io.create_writer_of_type` or get a list of all built-in readers with
        :meth:`~audiomate.corpus.io.available_writers`.

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
        raise NotImplementedError()
