import collections
import os
import glob
import re

import audiomate
from audiomate import annotations
from audiomate import issuers
from audiomate.corpus.subset import subview
from . import base
from . import downloader

DOWNLOAD_URL = 'http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/german-speechdata-package-v3.tar.gz'


SPEAKER_IDX_PATTERN = re.compile(r'<speaker_id>(.*?)</speaker_id>')
GENDER_PATTERN = re.compile(r'<gender>(.*?)</gender>')
TRANSCRIPTION_PATTERN = re.compile(r'<cleaned_sentence>(.*?)</cleaned_sentence>')
RAW_TRANSCRIPTION_PATTERN = re.compile(r'<sentence>(.*?)</sentence>')
AGE_PATTERN = re.compile(r'<ageclass>(.*?)</ageclass>')
NATIVE_PATTERN = re.compile(r'<muttersprachler>(.*?)</muttersprachler>')

SUBSETS = ['train', 'dev', 'test']

WAV_FILE_SUFFIXES = [
    'Kinect-Beam',
    'Kinect-RAW',
    'Realtek',
    'Samson',
    'Yamaha',
    'Microsoft-Kinect-Raw'
]

WAV_SUFFIX_TO_SUBVIEW = {
    'Kinect-Beam': 'kinect-beam',
    'Kinect-RAW': 'kinect-raw',
    'Realtek': 'realtek',
    'Samson': 'samson',
    'Yamaha': 'yamaha',
    'Microsoft-Kinect-Raw': 'kinect-raw',
}


class TudaDownloader(downloader.ArchiveDownloader):
    """
    Downloader for the TUDA Corpus.

    Args:
        url (str): The url to download the dataset from. If not given the
                   default URL is used. It is expected to be a tar.gz file.
        num_threads (int): Number of threads to use for download files.
    """

    def __init__(self, url=None, num_threads=1):
        if url is None:
            url = DOWNLOAD_URL

        super(TudaDownloader, self).__init__(
            url,
            move_files_up=True,
            num_threads=num_threads
        )

    @classmethod
    def type(cls):
        return 'tuda'


class TudaReader(base.CorpusReader):
    """
    Reader for the TUDA german distant speech corpus.

    Note:
        It only loads files ending in -beamformedSignal.wav

    .. seealso::

       `<https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/acoustic-models.html>`_
          Download page
    """

    @classmethod
    def type(cls):
        return 'tuda'

    def _check_for_missing_files(self, path):
        return []

    def _load(self, path):
        corpus = audiomate.Corpus(path=path)

        for part in SUBSETS:
            sub_path = os.path.join(path, part)
            ids = self.get_ids_from_folder(sub_path)
            utt_ids = []

            for idx in ids:
                add_ids = self.load_file(sub_path, idx, corpus)
                utt_ids.extend(add_ids)

            subview_filter = subview.MatchingUtteranceIdxFilter(utterance_idxs=utt_ids)
            subview_corpus = subview.Subview(corpus, filter_criteria=[subview_filter])
            corpus.import_subview(part, subview_corpus)

            TudaReader.create_wav_type_subviews(corpus, utt_ids, prefix='{}_'.format(part))

        TudaReader.create_wav_type_subviews(corpus, corpus.utterances.keys())

        return corpus

    @staticmethod
    def create_wav_type_subviews(corpus, utt_ids, prefix=''):
        splits = collections.defaultdict(list)

        for utt_id in utt_ids:
            wavtype = utt_id.split('_')[-1]
            subview_type = WAV_SUFFIX_TO_SUBVIEW[wavtype]
            splits[subview_type].append(utt_id)

        for sub_name, sub_utts in splits.items():
            subview_filter = subview.MatchingUtteranceIdxFilter(utterance_idxs=sub_utts)
            subview_corpus = subview.Subview(corpus, filter_criteria=[subview_filter])
            corpus.import_subview('{}{}'.format(prefix, sub_name), subview_corpus)

    def get_ids_from_folder(self, path):
        """
        Return all ids from the given folder,
        which have a corresponding beamformedSignal file.
        """
        ids = set()

        for xml_file in glob.glob(os.path.join(path, '*.xml')):
            idx = os.path.splitext(os.path.basename(xml_file))[0]
            ids.add(idx)

        return ids

    def load_file(self, folder_path, idx, corpus):
        """
        Load speaker, file, utterance, labels
        for the file with the given id.
        """
        xml_path = os.path.join(folder_path, '{}.xml'.format(idx))
        wav_paths = []

        for wav_suffix in WAV_FILE_SUFFIXES:
            wav_path = os.path.join(folder_path, '{}_{}.wav'.format(idx, wav_suffix))
            wav_name = os.path.split(wav_path)[1]
            wav_idx = os.path.splitext(wav_name)[0]

            if os.path.isfile(wav_path) and wav_idx not in self.invalid_utterance_ids:
                wav_paths.append(wav_path)

        if len(wav_paths) == 0:
            return []

        with open(xml_path, 'r', encoding='utf-8') as f:
            text = f.read()

        transcription = TudaReader.extract_value(text, TRANSCRIPTION_PATTERN, 'transcription', xml_path)
        transcription_raw = TudaReader.extract_value(text, RAW_TRANSCRIPTION_PATTERN, 'raw_transcription', xml_path)
        gender = TudaReader.extract_value(text, GENDER_PATTERN, 'gender', xml_path)
        is_native = TudaReader.extract_value(text, NATIVE_PATTERN, 'native', xml_path)
        age_class = TudaReader.extract_value(text, AGE_PATTERN, 'age', xml_path)
        speaker_idx = TudaReader.extract_value(text, SPEAKER_IDX_PATTERN, 'speaker_idx', xml_path)

        if speaker_idx not in corpus.issuers.keys():
            start_age_class = int(age_class.split('-')[0])

            if start_age_class < 12:
                age_group = issuers.AgeGroup.CHILD
            elif start_age_class < 18:
                age_group = issuers.AgeGroup.YOUTH
            elif start_age_class < 65:
                age_group = issuers.AgeGroup.ADULT
            else:
                age_group = issuers.AgeGroup.SENIOR

            native_lang = None

            if is_native == 'Ja':
                native_lang = 'deu'

            issuer = issuers.Speaker(speaker_idx,
                                     gender=issuers.Gender(gender),
                                     age_group=age_group,
                                     native_language=native_lang)
            corpus.import_issuers(issuer)

        utt_ids = []

        for wav_path in wav_paths:
            wav_name = os.path.split(wav_path)[1]
            wav_idx = os.path.splitext(wav_name)[0]
            corpus.new_file(wav_path, wav_idx)
            utt = corpus.new_utterance(wav_idx, wav_idx, speaker_idx)
            utt.set_label_list(annotations.LabelList.create_single(
                transcription,
                idx=audiomate.corpus.LL_WORD_TRANSCRIPT
            ))
            utt.set_label_list(annotations.LabelList.create_single(
                transcription_raw,
                idx=audiomate.corpus.LL_WORD_TRANSCRIPT_RAW
            ))
            utt_ids.append(wav_idx)

        return utt_ids

    @staticmethod
    def extract_value(text, pattern, value, path):
        m = pattern.search(text)

        if m:
            return m.group(1)
        else:
            raise ValueError('Value {} not found in {}'.format(value, path))
