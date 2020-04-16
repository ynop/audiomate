import os
import shutil

import audiomate
from audiomate import annotations
from audiomate import issuers
from audiomate.corpus import subset
from audiomate.utils import download
from audiomate.utils import textfile

from . import base


SUBSETS = {
    'dev-clean': 'http://www.openslr.org/resources/12/dev-clean.tar.gz',
    'dev-other': 'http://www.openslr.org/resources/12/dev-other.tar.gz',
    'test-clean': 'http://www.openslr.org/resources/12/test-clean.tar.gz',
    'test-other': 'http://www.openslr.org/resources/12/test-other.tar.gz',
    'train-clean-100': 'http://www.openslr.org/resources/12/train-clean-100.tar.gz',
    'train-clean-360': 'http://www.openslr.org/resources/12/train-clean-360.tar.gz',
    'train-other-500': 'http://www.openslr.org/resources/12/train-other-500.tar.gz',
}


class LibriSpeechDownloader(base.CorpusDownloader):
    """
    Downloader for the LibriSpeech Dataset.

    Args:
        subsets (list): List of subsets to download.
                        If empty or ``None``, all subsets are downloaded.
        num_threads (int): Number of threads to use for download files.
        keep_archives (bool): If ``True``, keep downloaded archives after extraction.
    """

    def __init__(self, subsets=None, num_threads=1, keep_archives=False):
        self.subsets = subsets
        self.num_threads = num_threads
        self.keep_archives = keep_archives

    @classmethod
    def type(cls):
        return 'librispeech'

    def _download(self, target_path):
        if self.subsets is None or len(self.subsets) == 0:
            to_download = SUBSETS.keys()
        else:
            to_download = self.subsets

        os.makedirs(target_path, exist_ok=True)

        for subset_name in to_download:
            if subset_name in SUBSETS.keys():
                tmp_file = os.path.join(target_path, '{}.tar.gz'.format(subset_name))
                download.download_file(
                    SUBSETS[subset_name],
                    tmp_file,
                    num_threads=self.num_threads
                )
                download.extract_tar(tmp_file, target_path)
                extract_sub_path = os.path.join(target_path, 'LibriSpeech')

                for item in os.listdir(extract_sub_path):
                    item_path = os.path.join(extract_sub_path, item)
                    item_target_path = os.path.join(target_path, item)
                    shutil.move(item_path, item_target_path)

                shutil.rmtree('extract_sub_path', ignore_errors=True)

                if not self.keep_archives:
                    os.remove(tmp_file)


class LibriSpeechReader(base.CorpusReader):
    """
    Reader for the LibriSpeech Corpus.

    .. seealso::

       `LibriSpeech <https://www.openslr.org/12/>`_
          Project Page
    """

    @classmethod
    def type(cls):
        return 'librispeech'

    def _check_for_missing_files(self, path):
        files = []

        for name in ['SPEAKERS.TXT']:
            if not os.path.isfile(os.path.join(path, name)):
                files.append(name)

        return files

    def _load(self, path):
        corpus = audiomate.Corpus(path=path)

        speaker_info_path = os.path.join(path, 'SPEAKERS.TXT')
        speakers = LibriSpeechReader.load_speakers(speaker_info_path)

        sf = LibriSpeechReader.available_subfolders

        for subset_idx, subset_path in sf(path, SUBSETS.keys()).items():
            subset_utt_ids = set()

            for speaker_idx, speaker_path in sf(subset_path).items():
                corpus.import_issuers(speakers[speaker_idx])

                for chapter_idx, chapter_path in sf(speaker_path).items():
                    transcript_path = os.path.join(
                        chapter_path,
                        '{}-{}.trans.txt'.format(speaker_idx, chapter_idx)
                    )
                    transcripts = LibriSpeechReader.load_transcripts(transcript_path)

                    for utt_idx, transcript in transcripts.items():
                        file_path = os.path.join(chapter_path, '{}.flac'.format(utt_idx))
                        corpus.new_file(file_path, utt_idx)

                        utterance = corpus.new_utterance(
                            utt_idx,
                            utt_idx,
                            speaker_idx
                        )

                        utterance.set_label_list(
                            annotations.LabelList.create_single(
                                transcript,
                                idx=audiomate.corpus.LL_WORD_TRANSCRIPT
                            )
                        )

                        subset_utt_ids.add(utt_idx)

            utt_filter = subset.MatchingUtteranceIdxFilter(utterance_idxs=set(subset_utt_ids))
            subview = subset.Subview(corpus, filter_criteria=[utt_filter])
            corpus.import_subview(subset_idx, subview)

        return corpus

    @staticmethod
    def available_subfolders(path, restrict=None):
        subfolders = {}

        for subfolder_name in os.listdir(path):
            if restrict is None or subfolder_name in restrict:
                subfolder_path = os.path.join(path, subfolder_name)

                if os.path.isdir(subfolder_path):
                    subfolders[subfolder_name] = subfolder_path

        return subfolders

    @staticmethod
    def load_speakers(path):
        entries = textfile.read_separated_lines_generator(
            path,
            separator='|',
            max_columns=5,
            ignore_lines_starting_with=[';']
        )

        speakers = {}

        for item in entries:
            idx = item[0].strip()
            gender_str = item[1].strip()

            if gender_str == 'M':
                gender = issuers.Gender.MALE
            elif gender_str == 'F':
                gender = issuers.Gender.FEMALE
            else:
                gender = issuers.Gender.UNKNOWN

            issuer = issuers.Speaker(
                idx,
                gender=gender
            )

            speakers[idx] = issuer

        return speakers

    @staticmethod
    def load_transcripts(path):
        entries = textfile.read_separated_lines_generator(
            path,
            separator=' ',
            max_columns=2
        )

        return {x[0].strip(): x[1].strip() for x in entries}
