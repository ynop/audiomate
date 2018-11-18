import os
import glob

import audiomate
from audiomate import annotations
from audiomate import issuers
from audiomate.corpus import subset
from audiomate.utils import download
from audiomate.utils import files
from audiomate.utils import textfile
from . import base

DOWNLOAD_V1 = 'https://s3.us-east-2.amazonaws.com/common-voice-data-download/cv_corpus_v1.tar.gz'


class CommonVoiceDownloader(base.CorpusDownloader):
    """
    Downloader for the Common Voice Corpus.

    Args:
        url (str): The url to download the dataset from. If not given the default URL is used.
                   It is expected to be a tar.gz file.
    """

    def __init__(self, url=None):
        if url is None:
            self.url = DOWNLOAD_V1
        else:
            self.url = url

    @classmethod
    def type(cls):
        return 'common-voice'

    def _download(self, target_path):
        os.makedirs(target_path, exist_ok=True)
        tmp_file = os.path.join(target_path, 'tmp_ark.tar.gz')

        download.download_file(self.url, tmp_file)
        download.extract_tar(tmp_file, target_path)

        files.move_all_files_from_subfolders_to_top(target_path, delete_subfolders=True)

        os.remove(tmp_file)


class CommonVoiceReader(base.CorpusReader):
    """
    Reader for the Common Voice Corpus.

    .. seealso::

       `Common-Voice <https://voice.mozilla.org/>`_
          Project Page
    """

    @classmethod
    def type(cls):
        return 'common-voice'

    def _check_for_missing_files(self, path):
        return []

    def _load(self, path):
        corpus = audiomate.Corpus(path=path)
        subset_ids = CommonVoiceReader.get_subset_ids(path)

        for subset_idx in subset_ids:
            CommonVoiceReader.load_subset(corpus, path, subset_idx)

        return corpus

    @staticmethod
    def get_subset_ids(path):
        """ Return a list with ids of all available subsets (based on existing csv-files). """
        all = []

        for path in glob.glob(os.path.join(path, '*.csv')):
            file_name = os.path.split(path)[1]
            basename = os.path.splitext(file_name)[0]
            all.append(basename)

        return all

    @staticmethod
    def load_subset(corpus, path, subset_idx):
        """ Load subset into corpus. """
        csv_file = os.path.join(path, '{}.csv'.format(subset_idx))
        utt_ids = []

        for entry in textfile.read_separated_lines_generator(csv_file, separator=',', max_columns=8,
                                                             ignore_lines_starting_with=['filename']):
            rel_file_path = entry[0]
            filename = os.path.split(rel_file_path)[1]
            basename = os.path.splitext(filename)[0]
            transcription = entry[1]
            age = CommonVoiceReader.map_age(entry[4])
            gender = CommonVoiceReader.map_gender(entry[5])

            idx = '{}-{}'.format(subset_idx, basename)
            file_path = os.path.join(path, rel_file_path)

            corpus.new_file(file_path, idx)
            issuer = issuers.Speaker(idx, gender=gender, age_group=age)
            corpus.import_issuers(issuer)
            utterance = corpus.new_utterance(idx, idx, issuer.idx)
            utterance.set_label_list(
                annotations.LabelList.create_single(
                    transcription,
                    idx=audiomate.corpus.LL_WORD_TRANSCRIPT
                )
            )

            utt_ids.append(idx)

        filter = subset.MatchingUtteranceIdxFilter(utterance_idxs=set(utt_ids))
        subview = subset.Subview(corpus, filter_criteria=[filter])
        corpus.import_subview(subset_idx, subview)

    @staticmethod
    def map_age(age):
        """ Map age to correct age-group. """

        if age in [None, '']:
            return issuers.AgeGroup.UNKNOWN
        elif age == 'teens':
            return issuers.AgeGroup.YOUTH
        elif age in ['sixties', 'seventies', 'eighties', 'nineties']:
            return issuers.AgeGroup.SENIOR
        else:
            return issuers.AgeGroup.ADULT

    @staticmethod
    def map_gender(gender):
        """ Map gender to correct value. """

        if gender == 'male':
            return issuers.Gender.MALE
        elif gender == 'female':
            return issuers.Gender.FEMALE
        else:
            return issuers.Gender.UNKNOWN
