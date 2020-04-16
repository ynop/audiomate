import abc
import copy
import os

import audiomate
from audiomate import tracks
from audiomate import logutil

logger = logutil.getLogger()


class AudioFileConverter(metaclass=abc.ABCMeta):
    """
    Base class for converters that convert all audio to a specific format.
    A converter creates a new instance of a corpus,
    so that all audio files meet given requirements.

    Args:
        sampling_rate (int): Target sampling rate to convert audio to.
        separate_file_per_utterance (bool): If ``True``, every utterance in the
                                       resulting corpus is in a separate file.
                                       If ``False``, the file/utt structure will
                                       be preserved.
        force_conversion (bool): If ``True``, all utterances will be converted
                                 whether or not it already matches the target
                                 format. If ``False``, only utterances not
                                 matching the target format will be converted.
                                 Others are reference to the original files.
    """

    def __init__(self, sampling_rate=16000, separate_file_per_utterance=False,
                 force_conversion=False):
        self.sampling_rate = sampling_rate
        self.separate_file_per_utterance = separate_file_per_utterance
        self.force_conversion = force_conversion

    def convert(self, corpus, target_audio_path):
        """
        Convert the given corpus.

        Args:
            corpus (Corpus): The input corpus.
            target_audio_path (str): The path where the audio files of the
                                     converted corpus should be saved.

        Returns:
            Corpus: The newly created corpus.
        """

        out_corpus = audiomate.Corpus()
        files_to_convert = []

        for utterance in logger.progress(
                corpus.utterances.values(),
                total=corpus.num_utterances,
                description='Find utterances to convert'):

            if utterance.issuer.idx not in out_corpus.issuers.keys():
                out_corpus.import_issuers(utterance.issuer)

            if self._does_utt_need_conversion(utterance):
                # Store audio in a new file

                if self.separate_file_per_utterance:
                    filename = '{}.{}'.format(utterance.idx, self._file_extension())
                    path = os.path.join(target_audio_path, filename)
                    files_to_convert.append((
                        utterance.track.path,
                        utterance.start,
                        utterance.end,
                        path
                    ))

                    track = out_corpus.new_file(path, utterance.idx)
                    start = 0
                    end = float('inf')

                else:
                    if utterance.track.idx not in out_corpus.tracks.keys():
                        filename = '{}.{}'.format(utterance.track.idx, self._file_extension())
                        path = os.path.join(target_audio_path, filename)
                        files_to_convert.append((
                            utterance.track.path,
                            0,
                            float('inf'),
                            path
                        ))
                        out_corpus.new_file(path, utterance.track.idx)

                    track = utterance.track
                    start = utterance.start
                    end = utterance.end

                utt = out_corpus.new_utterance(
                    utterance.idx,
                    track.idx,
                    issuer_idx=utterance.issuer.idx,
                    start=start,
                    end=end
                )

                lls = copy.deepcopy(list(utterance.label_lists.values()))
                utt.set_label_list(lls)

            else:
                # Just copy everything to the output corpus
                self._copy_utterance_to_corpus(utterance, out_corpus)

        self._copy_subviews_to_corpus(corpus, out_corpus)
        self._convert_files(files_to_convert)

        return out_corpus

    @abc.abstractmethod
    def _file_extension(self):
        """ Return the file-extension that will be used. """
        raise NotImplementedError()

    @abc.abstractmethod
    def _does_utt_match_target_format(self, utterance):
        """
        Return ``True`` if the utterance already matches the target format,
        ``False`` otherwise.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _convert_files(self, files):
        """
        Store the given samples with the target format
        at ``path``.
        """
        raise NotImplementedError()

    def _does_utt_need_conversion(self, utterance):
        """ Return True if an utterance needs to be converted. """
        if self.force_conversion:
            return True

        elif type(utterance.track) != tracks.FileTrack:
            return True

        elif self.separate_file_per_utterance and (utterance.start > 0 or utterance.end != float('inf')):
            return True

        elif not self._does_utt_match_target_format(utterance):
            return True

        return False

    def _copy_utterance_to_corpus(self, utterance, corpus):
        """ Create a copy of the utterance and add it to the given corpus. """

        if utterance.track.idx not in corpus.tracks.keys():
            corpus.import_tracks(utterance.track)

        corpus.import_utterances(utterance)

    def _copy_subviews_to_corpus(self, from_corpus, to_corpus):
        """ Create copy of all subviews from ``from_corpus`` in ``to_corpus``. """

        subviews = copy.deepcopy(from_corpus.subviews)
        for subview_idx, subview in subviews.items():
            to_corpus.import_subview(subview_idx, subview)
