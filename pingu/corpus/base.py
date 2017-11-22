import collections
import abc


class CorpusView(metaclass=abc.ABCMeta):
    """
    This class defines the basic interface of a corpus. It is not meant to be instantiated directly.
    It only describes the methods for accessing data of the corpus.

    Notes:
        All paths to files should be held as absolute paths in memory.

    Attributes:
        files (dict): A dictionary containing :py:class:`pingu.corpus.assets.File` objects with the file-idx as key.
        utterances (dict): A dictionary containing :py:class:`pingu.corpus.assets.Utterance` objects with the utterance-idx as key.
        issuers (dict): A dictionary containing :py:class:`pingu.corpus.assets.Issuer` objects with the issuer-idx as key.
        label_lists (dict): A dictionary containing utterance-idx/label_list dictionaries with the label-list-idx as key.
        feature_containers (dict): A dictionary containing :py:class:`pingu.corpus.assets.FeatureContainer` objects with the feature-idx as key.
    """

    def __init__(self):
        self.files = {}
        self.utterances = {}
        self.issuers = {}
        self.label_lists = collections.defaultdict(dict)
        self.feature_containers = {}

    @property
    @abc.abstractmethod
    def name(self):
        """ Return the name of the dataset (Equals basename of the path, if not None). """
        return "undefined"

    #
    #   Files
    #

    @property
    def num_files(self):
        """ Return number of files. """
        return len(self.files)

    #
    #   Utterances
    #

    @property
    def num_utterances(self):
        """ Return number of utterances. """
        return len(self.utterances)

    #
    #   Issuers
    #

    @property
    def num_issuers(self):
        """ Return the number of issuers in the dataset. """
        return len(self.issuers)

    #
    #   Feature Container
    #

    @property
    def num_feature_containers(self):
        """ Return the number of feature-containers in the dataset. """
        return len(self.feature_containers)
