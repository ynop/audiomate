import collections
import abc


class CorpusView(metaclass=abc.ABCMeta):
    """
    This class defines the basic interface of a corpus. It is not meant to be instantiated directly.
    It only describes the methods for accessing data of the corpus.

    Notes:
        All paths to files should be held as absolute paths in memory.
    """

    @property
    @abc.abstractmethod
    def name(self):
        """ Return the name of the dataset (Equals basename of the path, if not None). """
        return "undefined"

    #
    #   Files
    #

    @property
    @abc.abstractmethod
    def files(self):
        """
        Return the files in the corpus.

        Returns:
            dict: A dictionary containing :py:class:`pingu.corpus.assets.File` objects with the file-idx as key.
        """
        return {}

    @property
    def num_files(self):
        """ Return number of files. """
        return len(self.files)

    #
    #   Utterances
    #

    @property
    @abc.abstractmethod
    def utterances(self):
        """
        Return the utterances in the corpus.

        Returns:
            dict: A dictionary containing :py:class:`pingu.corpus.assets.Utterance` objects with the utterance-idx as key.
        """
        return {}

    @property
    def num_utterances(self):
        """ Return number of utterances. """
        return len(self.utterances)

    #
    #   Issuers
    #

    @property
    @abc.abstractmethod
    def issuers(self):
        """
        Return the issuers in the corpus.

        Returns:
            dict: A dictionary containing :py:class:`pingu.corpus.assets.Issuer` objects with the issuer-idx as key.
        """
        return {}

    @property
    def num_issuers(self):
        """ Return the number of issuers in the dataset. """
        return len(self.issuers)

    #
    #   Label List
    #

    @property
    @abc.abstractmethod
    def label_lists(self):
        """
        Return the label-lists in the corpus.

        Returns:
            dict: A dictionary containing utterance-idx/label_list dictionaries with the label-list-idx as key.
        """
        return collections.defaultdict(dict)

    #
    #   Feature Container
    #

    @property
    @abc.abstractmethod
    def feature_containers(self):
        """
        Return the feature-containers in the corpus.

        Returns:
            dict: A dictionary containing :py:class:`pingu.corpus.assets.FeatureContainer` objects with the feature-idx as key.
        """
        return {}

    @property
    def num_feature_containers(self):
        """ Return the number of feature-containers in the dataset. """
        return len(self.feature_containers)
