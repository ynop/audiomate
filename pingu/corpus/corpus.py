import os
import collections
import shutil

from pingu.corpus import assets
from pingu.utils import naming
from . import base

DEFAULT_FILE_SUBDIR = 'files'
DEFAULT_FEAT_SUBDIR = 'features'


class Corpus(base.CorpusView):
    """
    The Corpus class represents a single corpus.
    It extends :py:class:`pingu.corpus.CorpusView` with the functionality for loading and saving.
    Furthermore it provides the functionality for adding/modifying assets of the corpus like files and utterances.

    Args:
        path (str): Path where the corpus is stored. (Optional)
        loader (CorpusLoader): A loader to use for loading/saving. (By default the DefaultLoader is used)
    """

    def __init__(self, path=None, loader=None):
        super(Corpus, self).__init__()

        self.path = path

        # Set default loader of none is given
        if loader is None:
            from . import io
            self.loader = io.DefaultLoader()
        else:
            self.loader = loader

        self._files = {}
        self._utterances = {}
        self._issuers = {}
        self._label_lists = collections.defaultdict(dict)
        self._feature_containers = {}

    @property
    def name(self):
        if self.path is None:
            return "undefined"
        else:
            return os.path.basename(os.path.abspath(self.path))

    @property
    def files(self):
        return self._files

    @property
    def utterances(self):
        return self._utterances

    @property
    def issuers(self):
        return self._issuers

    @property
    def label_lists(self):
        return self._label_lists

    @property
    def feature_containers(self):
        return self._feature_containers

    #
    #   IO
    #

    def save(self):
        """
        If self.path is defined, it tries to save the corpus at the given path.
        """

        if self.path is None:
            raise ValueError('No path given to save the dataset.')

        self.save_at(self.path)

    def save_at(self, path, loader=None):
        """
        Save this corpus at the given path. If the path differs from the current path set, the path gets updated.

        Parameters:
            path (str): Path to save the dataset to.
            loader (str, CorpusLoader): If you want to use another loader (e.g. to export to another format). Otherwise it uses the loader associated with this dataset.
        """

        if loader is None:
            # If not loader given, use the one associated with the corpus
            self.loader.save(self, path)

        elif type(loader) == str:
            # If a loader is given as string, try to create such a loader.
            from . import io
            loader = io.create_loader_of_type(loader)
            loader.save(self, path)

        else:
            # Use the given loader
            loader.save(self, path)

        self.path = path

    @classmethod
    def load(cls, path, loader=None):
        """
        Loads the corpus from the given path, using the given loader. If no loader is given the default loader is used.

        Args:
            path (str): Path to load the corpus from.
            loader (str, CorpusLoader): The loader or the name of the loader to use.

        Returns:
            Corpus: The loaded corpus.
        """

        if loader is None:
            from . import io
            loader = io.DefaultLoader()

        elif type(loader) == str:
            from . import io
            loader = io.create_loader_of_type(loader)

        return loader.load(path)

    #
    # File
    #

    def new_file(self, path, file_idx, copy_file=False):
        """
        Adds a new file to the dataset with the given data.

        Parameters:
            path (str): Path of the file to add.
            file_idx (str): The id to associate the file with.
            copy_file (bool): If True the file is copied to the dataset folder, otherwise the given path is used directly.

        Returns:
            File: The newly added File.
        """

        new_file_idx = file_idx
        new_file_path = os.path.abspath(path)

        # Add index to idx if already existing
        if new_file_idx in self._files.keys():
            new_file_idx = naming.index_name_if_in_list(new_file_idx, self._files.keys())

        # Copy file to default file dir
        if copy_file:
            if not os.path.isdir(self.path):
                raise ValueError('To copy file the dataset needs to have a path.')

            __, ext = os.path.splitext(path)

            new_file_folder = os.path.join(self.path, DEFAULT_FILE_SUBDIR)
            new_file_path = os.path.join(new_file_folder, '{}{}'.format(new_file_idx, ext))
            os.makedirs(new_file_folder, exist_ok=True)
            shutil.copy(path, new_file_path)

        # Create file obj
        new_file = assets.File(new_file_idx, new_file_path)
        self._files[new_file_idx] = new_file

        return new_file

    #
    #   Utterances
    #

    def new_utterance(self, utterance_idx, file_idx, issuer_idx=None, start=0, end=-1):
        """
        Add a new utterance to the dataset with the given data.

        Parameters:
            file_idx (str): The file id the utterance is in.
            utterance_idx (str): The id to associate with the utterance. If None or already exists, one is generated.
            issuer_idx (str): The issuer id to associate with the utterance.
            start (float): Start of the utterance within the file [seconds].
            end (float): End of the utterance within the file [seconds]. -1 equals the end of the file.

        Returns:
            Utterance: The newly added utterance.
        """

        new_utt_idx = utterance_idx

        # Check if there is a file with the given idx
        if not file_idx in self._files.keys():
            raise ValueError('No file in dataset with id {} to add utterance.'.format(file_idx))

        # Add index to idx if already existing
        if new_utt_idx in self._utterances.keys():
            new_utt_idx = naming.index_name_if_in_list(new_utt_idx, self._utterances.keys())

        new_utt = assets.Utterance(new_utt_idx, file_idx, issuer_idx=issuer_idx, start=start, end=end)
        self._utterances[new_utt_idx] = new_utt

        return new_utt

    #
    #   Issuer
    #

    def new_issuer(self, issuer_idx, info=None):
        """
        Add a new issuer to the dataset with the given data.

        Parameters:
            issuer_idx (str): The id to associate the issuer with. If None or already exists, one is generated.
            info (dict, list): Additional info of the issuer.

        Returns:
            Issuer: The newly added issuer.
        """

        new_issuer_idx = issuer_idx

        # Add index to idx if already existing
        if new_issuer_idx in self._issuers.keys():
            new_issuer_idx = naming.index_name_if_in_list(new_issuer_idx, self._issuers.keys())

        new_issuer = assets.Issuer(new_issuer_idx, info=info)
        self._issuers[new_issuer_idx] = new_issuer

        return new_issuer

    #
    #   Labeling
    #

    def new_label_list(self, utterance_idx, idx=None, labels=None):
        """
        Add a new label-list with the given data.

        Parameters:
            utterance_idx (str): Utterance id the label-list is associated with.
            idx (str): An identifier this label-list is associated with.
            labels (list): Labels to add to the new label-list.

        Returns:
            LabelList: The newly added label-list.
        """

        new_label_list_idx = idx

        if new_label_list_idx is None:
            new_label_list_idx = 'default'

        new_label_list = assets.LabelList(idx=new_label_list_idx)

        if isinstance(labels, assets.Label):
            new_label_list.append(labels)
        elif isinstance(labels, list):
            new_label_list.extend(labels)

        self._label_lists[new_label_list.idx][utterance_idx] = new_label_list

        return new_label_list

    #
    #   FEATURES
    #

    def new_feature_container(self, idx, path=None):
        """
        Add a new feature container with the given data.

        Parameters:
            idx (str): An unique identifier within the dataset.
            path (str): The path to store the feature file. If None a default path is used.

        Returns:
            FeatureContainer: The newly added feature-container.
        """

        new_feature_idx = idx
        new_feature_path = path

        # Add index to idx if already existing
        if new_feature_idx in self._feature_containers.keys():
            new_feature_idx = naming.index_name_if_in_list(new_feature_idx, self._feature_containers.keys())

        # Set default path if none given
        if new_feature_path is None:
            if not os.path.isdir(self.path):
                raise ValueError('To copy file the dataset needs to have a path.')

            new_feature_path = os.path.join(self.path, DEFAULT_FEAT_SUBDIR, new_feature_idx)
        else:
            new_feature_path = os.path.abspath(new_feature_path)

        container = assets.FeatureContainer(new_feature_path)
        self._feature_containers[new_feature_idx] = container

        return container
