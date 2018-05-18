import os
import glob

import audiomate
from . import base


class FolderReader(base.CorpusReader):
    """
    Loads all wavs from the given folder and creates a corpus from it.
    """

    @classmethod
    def type(cls):
        return 'folder'

    def _check_for_missing_files(self, path):
        return []

    def _load(self, path):
        corpus = audiomate.Corpus(path=path)

        for item in glob.glob('{}/*.wav'.format(path), recursive=True):
            basename, __ = os.path.splitext(os.path.basename(item))
            corpus.new_file(item, basename)
            corpus.new_utterance(basename, basename)

        return corpus
