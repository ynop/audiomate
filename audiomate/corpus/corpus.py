import copy
import os
import shutil

from audiomate.corpus import assets
from audiomate.utils import naming
from . import base
from . import subset

DEFAULT_FILE_SUBDIR = 'files'
DEFAULT_FEAT_SUBDIR = 'features'


class Corpus(base.CorpusView):
    """
    The Corpus class represents a single corpus.
    It extends :py:class:`audiomate.corpus.CorpusView` with the functionality for loading and saving.
    Furthermore it provides the functionality for adding/modifying assets of the corpus like files
    and utterances.

    Args:
        path (str): Path where the corpus is stored. (Optional)
    """

    def __init__(self, path=None):
        super(Corpus, self).__init__()

        self.path = path
        self._files = {}
        self._utterances = {}
        self._issuers = {}
        self._feature_containers = {}
        self._subviews = {}

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
    def feature_containers(self):
        return self._feature_containers

    @property
    def subviews(self):
        return self._subviews

    #
    #   IO
    #

    def save(self, writer=None):
        """
        If self.path is defined, it tries to save the corpus at the given path.
        """

        if self.path is None:
            raise ValueError('No path given to save the data set.')

        self.save_at(self.path, writer)

    def save_at(self, path, writer=None):
        """
        Save this corpus at the given path. If the path differs from the current path set, the path
        gets updated.

        Parameters:
            path (str): Path to save the data set to.
            writer (str, CorpusWriter): The writer or the name of the reader to use.
        """

        if writer is None:
            from . import io
            writer = io.DefaultWriter()
        elif type(writer) == str:
            # If a loader is given as string, try to create such a loader.
            from . import io
            writer = io.create_writer_of_type(writer)

        writer.save(self, path)

        self.path = path

    @classmethod
    def load(cls, path, reader=None):
        """
        Loads the corpus from the given path, using the given reader. If no reader is given the
        :py:class:`audiomate.corpus.io.DefaultReader` is used.

        Args:
            path (str): Path to load the corpus from.
            reader (str, CorpusReader): The reader or the name of the reader to use.

        Returns:
            Corpus: The loaded corpus.
        """

        if reader is None:
            from . import io
            reader = io.DefaultReader()

        elif type(reader) == str:
            from . import io
            reader = io.create_reader_of_type(reader)

        return reader.load(path)

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
            files (list): Either a list of or a single :py:class:`audiomate.corpus.assets.File`.

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
            raise ValueError('File with id {} does not exist!'.format(file_idx))

        # Check if issuer exists
        issuer = None

        if issuer_idx is not None:
            if issuer_idx not in self._issuers.keys():
                raise ValueError('Issuer with id {} does not exist!'.format(issuer_idx))
            else:
                issuer = self._issuers[issuer_idx]

        # Add index to idx if already existing
        if new_utt_idx in self._utterances.keys():
            new_utt_idx = naming.index_name_if_in_list(new_utt_idx, self._utterances.keys())

        new_utt = assets.Utterance(new_utt_idx,
                                   self.files[file_idx],
                                   issuer=issuer,
                                   start=start,
                                   end=end)

        self._utterances[new_utt_idx] = new_utt

        return new_utt

    def import_utterances(self, utterances):
        """
        Add the given utterances/utterance to the corpus.
        If any of the given utterance-ids already exists, a suffix is appended so it is unique.

        Args:
            utterances (list): Either a list of or a single :py:class:`audiomate.corpus.assets.Utterance`.

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
            if utterance.file not in self._files.values():
                raise ValueError('File with id {} is not in the corpus.'.format(utterance.file.idx, utterance.idx))

            # Check if there is a issuer with the given idx
            if utterance.issuer is not None and utterance.issuer not in self._issuers.values():
                raise ValueError('No issuer in corpus with id {} to add utterance {}.'.format(
                    utterance.issuer.idx, utterance.idx))

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
            issuers (list): Either a list of or a single :py:class:`audiomate.corpus.assets.Issuer`.

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
    #   Subviews
    #

    def import_subview(self, idx, subview):
        """
        Add the given subview to the corpus.

        Args:
            idx (str): An idx that is unique in the corpus for identifying the subview.
                       If already a subview exists with the given id it will be overridden.
            subview (Subview): The subview to add.
        """

        subview.corpus = self
        self._subviews[idx] = subview

    #
    #   Merge
    #

    def merge_corpus(self, corpus):
        """
        Merge the given corpus into this corpus. All assets (files, utterances, issuers, ...) are copied into
        this corpus. If any ids (utt-idx, file-idx, issuer-idx, subview-idx, ...) are occurring in both corpora,
        the ids from the merging corpus are suffixed by a number (starting from 1 until no other is matching).

        Args:
            corpus (CorpusView): The corpus to merge.
        """

        # Create a copy, so objects aren't changed in the original merging corpus
        merging_corpus = Corpus.from_corpus(corpus)

        self.import_files(corpus.files.values())
        self.import_issuers(corpus.issuers.values())
        utterance_idx_mapping = self.import_utterances(corpus.utterances.values())

        for subview_idx, subview in merging_corpus.subviews.items():
            for filter in subview.filter_criteria:
                if isinstance(filter, subset.MatchingUtteranceIdxFilter):
                    new_filtered_utt_ids = set()
                    for utt_idx in filter.utterance_idxs:
                        new_filtered_utt_ids.add(utterance_idx_mapping[utt_idx].idx)
                    filter.utterance_idxs = new_filtered_utt_ids

            new_idx = naming.index_name_if_in_list(subview_idx, self.subviews.keys())
            self.import_subview(new_idx, subview)

        for feat_container_idx, feat_container in merging_corpus.feature_containers.items():
            self.new_feature_container(feat_container_idx, feat_container.path)

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
            utterance.file = file_mapping[utterance.file.idx]

            if utterance.issuer is not None:
                utterance.issuer = issuer_mapping[utterance.issuer.idx]

        ds.import_utterances(utterances)

        # Subviews
        subviews = copy.deepcopy(corpus.subviews)
        for subview_idx, subview in subviews.items():
            ds.import_subview(subview_idx, subview)

        # Feat-Containers
        for feat_container_idx, feature_container in corpus.feature_containers.items():
            ds.new_feature_container(feat_container_idx, feature_container.path)

        return ds

    @classmethod
    def merge_corpora(cls, corpora):
        """
        Merge a list of corpora into one.

        Args:
            corpora (Iterable): An iterable of :py:class:`audiomate.corpus.CorpusView`.

        Returns:
            Corpus: A corpus with the data from all given corpora merged into one.
        """

        ds = Corpus()

        for merging_corpus in corpora:
            ds.merge_corpus(merging_corpus)

        return ds
