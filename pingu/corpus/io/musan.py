import os

import pingu
from pingu.corpus import assets
from pingu.utils import textfile
from . import base

AUDIO_TYPES_ = ['music', 'noise', 'speech']

ANN_NUM_COLUMS_ = {'music': 4, 'noise': 1, 'speech': 3}

ANN_FILE_NAME_ = 'ANNOTATIONS'


class MusanLoader(base.CorpusLoader):
    """
    Corpus loader for the MUSAN corpus. MUSAN is a corpus of music, speech, and noise recordings.

    # References
        - [MUSAN: A Music, Speech, and Noise Corpus](https://arxiv.org/pdf/1510.08484v1.pdf)
        - [OpenSLR: MUSAN](http://www.openslr.org/17/) (download)
    """

    @classmethod
    def type(cls):
        return 'musan'

    def _check_for_missing_files(self, path):
        # Some annotation files are missing anyway in the original dataset (e.g. speech/us-gov/ANNOTATIONS).
        # What's left would be checking for missing directories.
        pass

    def _load(self, path):
        create_or_get_issuer = {
            'music': self._create_or_get_music_issuer,
            'noise': self._create_or_get_noise_issuer,
            'speech': self._create_or_get_speech_issuer,
        }

        corpus = pingu.Corpus(path=path, loader=self)

        for type_name, type_directory in self._directories(path).items():
            for _, source_directory in self._directories(type_directory).items():
                annotations_path = os.path.join(source_directory, ANN_FILE_NAME_)
                annotations = {}

                if os.path.exists(annotations_path):
                    annotations = textfile.read_separated_lines_with_first_key(annotations_path, separator=' ',
                                                                               max_columns=ANN_NUM_COLUMS_[type_name])

                with os.scandir(source_directory) as it:
                    for entry in it:
                        if not entry.name.endswith('.wav'):
                            continue

                        file_path = os.path.join(source_directory, entry.name)
                        file_idx = entry.name[0:-4]  # chop of .wav
                        utterance_idx = file_idx  # every file is a separate utterance
                        issuer_idx = create_or_get_issuer[type_name](corpus, file_idx, annotations)

                        corpus.new_file(file_path, file_idx=file_idx, copy_file=False)
                        corpus.new_utterance(utterance_idx, file_idx, issuer_idx)
                        corpus.new_label_list(utterance_idx, idx='audio_type', labels=[assets.Label(type_name)])

        return corpus

    def _save(self, corpus, path):
        raise NotImplementedError("There is no implementation for saving in MUSAN format.")

    @staticmethod
    def _directories(path):
        directories = {}
        with os.scandir(path) as it:
            for entry in it:
                if not entry.is_dir():
                    continue

                directories[entry.name] = os.path.join(path, entry.name)

        return directories

    # noinspection PyUnusedLocal
    @staticmethod
    def _create_or_get_noise_issuer(corpus, file_idx, annotations):
        return None

    @staticmethod
    def _create_or_get_music_issuer(corpus, file_idx, annotations):
        if file_idx not in annotations:
            return None

        issuer_idx = annotations[file_idx][2]

        if issuer_idx not in corpus.issuers:
            corpus.new_issuer(issuer_idx)

        return issuer_idx

    @staticmethod
    def _create_or_get_speech_issuer(corpus, file_idx, annotations):
        if file_idx not in annotations:
            return None

        if file_idx in annotations:
            corpus.new_issuer(file_idx, {'gender': annotations[file_idx][0], 'language': annotations[file_idx][1]})
        else:
            corpus.new_issuer(file_idx, {'gender': None, 'language': None})

        return file_idx
