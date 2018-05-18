import os

import audiomate
from audiomate.corpus import assets
from . import base

DIRECTORIES = {'music_wav': 'music', 'speech_wav': 'speech'}


class GtzanReader(base.CorpusReader):
    """
    Reader for the GTZAN music/speech corpus. The corpus consits of 64 music and 64 speech tracks that are each 30
    seconds long. The Wave files are 16-bit mono and have a sampling rate of 22050 Hz.

    .. seealso::

       `MARSYAS: GTZAN Music/Speech <https://marsyasweb.appspot.com/download/data_sets/>`_
          Download page
    """

    @classmethod
    def type(cls):
        return 'gtzan'

    def _check_for_missing_files(self, path):
        return []

    def _load(self, path):
        corpus = audiomate.Corpus(path=path)

        for directory, type_name in DIRECTORIES.items():
            source_directory = os.path.join(path, directory)

            if not os.path.isdir(source_directory):
                continue

            it = os.scandir(source_directory)

            for entry in it:
                if not entry.name.endswith('.wav'):
                    continue

                file_path = os.path.join(source_directory, entry.name)
                file_idx = entry.name[0:-4]  # chop of .wav
                utterance_idx = file_idx  # every file is a separate utterance

                corpus.new_file(file_path, file_idx=file_idx, copy_file=False)
                utterance = corpus.new_utterance(utterance_idx, file_idx)
                utterance.set_label_list(assets.LabelList(idx='audio_type', labels=[assets.Label(type_name)]))

        return corpus
