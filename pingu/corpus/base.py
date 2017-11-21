import collections
import abc


class CorpusView(metaclass=abc.ABCMeta):
    """
    This class defines the basic interface of a corpus. It is not meant to be instantiated directly.

    Attributes:
        files: A dictionary containing file-identifiers (keys) and file-paths (values). The file-paths are absolute paths when loaded in a dataset.
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
