import os

import audiomate
from audiomate import annotations
from audiomate import issuers
from audiomate import logutil

from . import base

logger = logutil.getLogger()


# ==================================================================================================

# See https://github.com/kyubyong/css10#audiobooks--datasets for download links
# (Kaggle account needed)

# ==================================================================================================


class CssTenReader(base.CorpusReader):
    """ Reader for collections of css10 audio data.
    The reader expects extracted .tgz files in the given folder. """

    @classmethod
    def type(cls):
        return "css10"

    def _check_for_missing_files(self, path):
        return []

    # ==============================================================================================

    def _load(self, path):
        # Create a new corpus
        corpus = audiomate.Corpus(path=path)

        # Get transcripts from file
        transcripts = self.read_transcripts(path)

        for t in logger.progress(transcripts):
            # Create files ...
            file_path = os.path.join(path, t[0])
            file_idx = os.path.splitext(os.path.basename(file_path))[0]
            corpus.new_file(file_path, file_idx)

            # Issuers, use folder name
            issuer_idx = t[0].split("/")[0]
            issuer = issuers.Speaker(issuer_idx)
            corpus.import_issuers(issuer)

            # Utterances with labels ...
            utterance = corpus.new_utterance(file_idx, file_idx, issuer_idx)
            ll_raw = annotations.LabelList.create_single(
                t[1], idx=audiomate.corpus.LL_WORD_TRANSCRIPT_RAW
            )
            ll_clean = annotations.LabelList.create_single(
                t[2], idx=audiomate.corpus.LL_WORD_TRANSCRIPT
            )
            utterance.set_label_list(ll_raw)
            utterance.set_label_list(ll_clean)

        return corpus

    # ==============================================================================================

    @staticmethod
    def read_transcripts(path):
        with open(os.path.join(path, "transcript.txt"), "r", encoding="utf-8") as file:
            content = file.readlines()

        transcripts = []
        for t in content:
            t = t.split("|")
            transcripts.append(t)

        return transcripts
