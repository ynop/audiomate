import glob
import os

import audiomate
from audiomate import annotations
from audiomate import issuers
from audiomate.corpus.subset import subview
from audiomate.utils import textfile
from . import base


class SpeechCommandsReader(base.CorpusReader):
    """
    Reads the google speech commands dataset.

    .. seealso::

        `Launching Speech Commands DS <https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html>`_
            Blog-Entry on the release of the speech commands dataset.

    """

    @classmethod
    def type(cls):
        return 'speech-commands'

    def _check_for_missing_files(self, path):
        return []

    def _load(self, path):
        corpus = audiomate.Corpus(path=path)

        for folder in os.scandir(path):
            if folder.is_dir() and not folder.name.startswith('_'):
                SpeechCommandsReader._load_folder(folder, corpus)
                SpeechCommandsReader._create_subviews(path, corpus)

        return corpus

    @staticmethod
    def _load_folder(folder_entry, corpus):
        """ Load the given subfolder into the corpus (e.g. bed, one, ...) """
        for wav_path in glob.glob(os.path.join(folder_entry.path, '*.wav')):
            wav_name = os.path.basename(wav_path)
            basename, __ = os.path.splitext(wav_name)

            command = folder_entry.name
            file_idx = '{}_{}'.format(basename, command)
            issuer_idx = str(basename).split('_', maxsplit=1)[0]

            corpus.new_file(wav_path, file_idx)

            if issuer_idx not in corpus.issuers.keys():
                corpus.import_issuers(issuers.Speaker(
                    issuer_idx
                ))

            utt = corpus.new_utterance(file_idx, file_idx, issuer_idx)

            labels = annotations.LabelList.create_single(command, idx=audiomate.corpus.LL_WORD_TRANSCRIPT)
            utt.set_label_list(labels)

    @staticmethod
    def _create_subviews(path, corpus):
        """ Load the subviews based on testing_list.txt and validation_list.txt """
        test_list_path = os.path.join(path, 'testing_list.txt')
        dev_list_path = os.path.join(path, 'validation_list.txt')

        test_list = textfile.read_separated_lines(test_list_path, separator='/', max_columns=2)
        dev_list = textfile.read_separated_lines(dev_list_path, separator='/', max_columns=2)

        test_set = set(['{}_{}'.format(os.path.splitext(x[1])[0], x[0]) for x in test_list])
        dev_set = set(['{}_{}'.format(os.path.splitext(x[1])[0], x[0]) for x in dev_list])
        inv_train_set = test_set.union(dev_set)

        train_filter = subview.MatchingUtteranceIdxFilter(utterance_idxs=inv_train_set, inverse=True)
        train_view = subview.Subview(corpus, filter_criteria=train_filter)
        corpus.import_subview('train', train_view)

        dev_filter = subview.MatchingUtteranceIdxFilter(utterance_idxs=dev_set, inverse=False)
        dev_view = subview.Subview(corpus, filter_criteria=dev_filter)
        corpus.import_subview('dev', dev_view)

        test_filter = subview.MatchingUtteranceIdxFilter(utterance_idxs=test_set, inverse=False)
        test_view = subview.Subview(corpus, filter_criteria=test_filter)
        corpus.import_subview('test', test_view)
