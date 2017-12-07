import collections
import copy
import os
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
    Furthermore it provides the functionality for adding/modifying assets of the corpus like files
    and utterances.

    Args:
        path (str): Path where the corpus is stored. (Optional)
        loader (CorpusLoader): A loader to use for loading/saving. (By default the DefaultLoader is
                               used)
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
            return 'undefined'
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
        Save this corpus at the given path. If the path differs from the current path set, the path
        gets updated.

        Parameters:
            path (str): Path to save the data set to.
            loader (str, CorpusLoader): If you want to use another loader (e.g. to export to another
                                        format). Otherwise it uses the loader associated with this
                                        data set.
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
        Loads the corpus from the given path, using the given loader. If no loader is given the
        default loader is used.

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
        Adds a new file to the corpus with the given data.

        Parameters:
            path (str): Path of the file to add.
            file_idx (str): The id to associate the file with.
            copy_file (bool): If True the file is copied to the data set folder, otherwise the given
                              path is used directly.

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

    def import_files(self, files):
        """
        Add the given files/file to the corpus.
        If any of the given file-ids already exists, a suffix is appended so it is unique.

        Args:
            files (list): Either a list of or a single :py:class:`pingu.corpus.assets.File`.

        Returns:
            dict: A dictionary containing file idx mappings (old-file-idx/file-instance).
                  If a file is imported, whose id already exists this mapping can be used to check
                  the new id.
        """

        if isinstance(files, assets.File):
            files = [files]

        idx_mapping = {}

        for file in files:
            idx_mapping[file.idx] = file

            # Add index to idx if already existing
            if file.idx in self._files.keys():
                file.idx = naming.index_name_if_in_list(file.idx, self._files.keys())

            self._files[file.idx] = file

        return idx_mapping

    #
    #   Utterances
    #

    def new_utterance(self, utterance_idx, file_idx, issuer_idx=None, start=0, end=-1):
        """
        Add a new utterance to the corpus with the given data.

        Parameters:
            file_idx (str): The file id the utterance is in.
            utterance_idx (str): The id to associate with the utterance. If None or already exists,
                                 one is generated.
            issuer_idx (str): The issuer id to associate with the utterance.
            start (float): Start of the utterance within the file [seconds].
            end (float): End of the utterance within the file [seconds]. -1 equals the end of the
                         file.

        Returns:
            Utterance: The newly added utterance.
        """

        new_utt_idx = utterance_idx

        # Check if there is a file with the given idx
        if file_idx not in self._files.keys():
            raise ValueError('No file in dataset with id {} to add utterance.'.format(file_idx))

        # Add index to idx if already existing
        if new_utt_idx in self._utterances.keys():
            new_utt_idx = naming.index_name_if_in_list(new_utt_idx, self._utterances.keys())

        new_utt = assets.Utterance(new_utt_idx, file_idx, issuer_idx=issuer_idx, start=start,
                                   end=end)
        self._utterances[new_utt_idx] = new_utt

        return new_utt

    def import_utterances(self, utterances):
        """
        Add the given utterances/utterance to the corpus.
        If any of the given utterance-ids already exists, a suffix is appended so it is unique.

        Args:
            utterances (list): Either a list of or a single
            :py:class:`pingu.corpus.assets.Utterance`.

        Returns:
            dict: A dictionary containing file idx mappings (old-utterance-idx/utterance-instance).
                  If a utterance is imported, whose id already exists this mapping can be used to
                  check the new id.
        """

        if isinstance(utterances, assets.Utterance):
            utterances = [utterances]

        idx_mapping = {}

        for utterance in utterances:
            idx_mapping[utterance.idx] = utterance

            # Check if there is a file with the given idx
            if utterance.file_idx not in self._files.keys():
                raise ValueError(
                    'No file in corpus with id {} to add utterance {}.'.format(utterance.file_idx,
                                                                               utterance.idx))

            # Check if there is a issuer with the given idx
            if utterance.issuer_idx is not None \
                    and utterance.issuer_idx not in self._issuers.keys():
                raise ValueError('No issuer in corpus with id {} to add utterance {}.'.format(
                    utterance.issuer_idx, utterance.idx))

            # Add index to idx if already existing
            if utterance.idx in self._utterances.keys():
                utterance.idx = naming.index_name_if_in_list(utterance.idx, self._utterances.keys())

            self._utterances[utterance.idx] = utterance

        return idx_mapping

    #
    #   Issuer
    #

    def new_issuer(self, issuer_idx, info=None):
        """
        Add a new issuer to the dataset with the given data.

        Parameters:
            issuer_idx (str): The id to associate the issuer with. If None or already exists, one is
                              generated.
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

    def import_issuers(self, issuers):
        """
        Add the given issuers/issuer to the corpus.
        If any of the given issuer-ids already exists, a suffix is appended so it is unique.

        Args:
            issuers (list): Either a list of or a single :py:class:`pingu.corpus.assets.Issuer`.

        Returns:
            dict: A dictionary containing file idx mappings (old-issuer-idx/issuer-instance).
                  If a issuer is imported, whose id already exists this mapping can be used to check
                  the new id.
        """

        if isinstance(issuers, assets.Issuer):
            issuers = [issuers]

        idx_mapping = {}

        for issuer in issuers:
            idx_mapping[issuer.idx] = issuer

            # Add index to idx if already existing
            if issuer.idx in self._issuers.keys():
                issuer.idx = naming.index_name_if_in_list(issuer.idx, self._issuers.keys())

            self._issuers[issuer.idx] = issuer

        return idx_mapping

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

    def import_label_list(self, utterance_idx, label_list):
        """
        Add the given label_list to the corpus for the given utterance-idx.
        If the label-list-id already exists, it is overridden

        Args:
            label_list (LabelList): A label-list
            utterance_idx (str): A utterance-id for which to add the given label-list
        """

        # Check if there is a utterance with the given idx
        if utterance_idx not in self._utterances.keys():
            raise ValueError(
                'No utterance in corpus with id {} to add label-list {}.'.format(utterance_idx,
                                                                                 label_list.idx))

        self._label_lists[label_list.idx][utterance_idx] = label_list

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
            new_feature_idx = naming.index_name_if_in_list(new_feature_idx,
                                                           self._feature_containers.keys())

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

    #
    #   Creation
    #

    @classmethod
    def from_corpus(cls, corpus):
        """
        Create a new modifiable corpus from any other CorpusView.
        This for example can be used to create a independent modifiable corpus from a subview.

        Args:
            corpus (CorpusView): The corpus to create a copy from.

        Returns:
            Corpus: A new corpus with the same data as the given one.
        """

        ds = Corpus()

        # Files
        files = copy.deepcopy(list(corpus.files.values()))
        file_mapping = ds.import_files(files)

        # Issuers
        issuers = copy.deepcopy(list(corpus.issuers.values()))
        issuer_mapping = ds.import_issuers(issuers)

        # Utterances, with replacing changed file- and issuer-ids
        utterances = copy.deepcopy(list(corpus.utterances.values()))
        for utterance in utterances:
            utterance.file_idx = file_mapping[utterance.file_idx].idx
            utterance.issuer_idx = issuer_mapping[utterance.issuer_idx].idx
        utterance_mapping = ds.import_utterances(utterances)

        # Label-lists
        for idx, label_lists in corpus.label_lists.items():
            for utt_idx, label_list in label_lists.items():
                new_utt_idx = utterance_mapping[utt_idx].idx
                ds.import_label_list(new_utt_idx, copy.deepcopy(label_list))

        return ds
