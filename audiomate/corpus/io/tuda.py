import collections
import os
import glob

from bs4 import BeautifulSoup

import audiomate
from audiomate import annotations
from audiomate import issuers
from audiomate.corpus.subset import subview
from . import base

SUBSETS = ['train', 'dev', 'test']

# Wrong transcripts, empty or to short
BAD_FILES = {
    'train': [
        # INVALID AUDIO
        '2014-03-18-15-29-23', '2014-03-18-15-28-52', '2014-03-24-13-39-24',
        '2014-08-05-11-08-34', '2014-03-27-11-50-33',

        # TO SHORT FOR THE TRANSCRIPTION
        '2014-08-04-13-09-09', '2014-08-04-13-14-27', '2014-08-04-13-39-33',
        '2014-03-20-10-44-06', '2014-08-04-13-39-55', '2014-08-04-13-23-36',
        '2014-08-04-13-13-57', '2014-08-04-13-15-41', '2014-08-04-13-39-11',
        '2014-08-04-13-05-54', '2014-08-04-13-05-57', '2014-05-13-11-42-53',
        '2014-08-04-13-07-47', '2014-08-04-13-08-49', '2014-08-04-13-11-42',
        '2014-08-04-13-08-28', '2014-08-04-13-11-36', '2014-08-04-13-16-45',
        '2014-03-17-13-06-10', '2014-08-04-13-14-38', '2014-03-17-13-07-50',
        '2014-08-27-11-05-29', '2014-08-04-13-06-26', '2014-08-04-13-07-42',
        '2014-08-04-13-08-45', '2014-03-27-10-47-21', '2014-06-17-13-46-27',
        '2014-03-17-13-16-59', '2014-03-17-13-09-27', '2014-08-04-13-37-33',
        '2014-08-04-13-15-34', '2014-08-04-13-15-45', '2014-08-04-13-06-01',
        '2014-08-04-13-04-58', '2014-08-04-13-16-29', '2014-08-04-13-08-53',
        '2014-08-04-13-21-42', '2014-08-04-13-40-11', '2014-08-04-13-15-20',
        '2014-03-17-13-03-26', '2014-08-04-13-21-50', '2014-08-04-13-05-35',
        '2014-08-04-13-22-57', '2014-08-04-13-22-17', '2014-08-04-13-39-21',
        '2014-08-04-13-21-58', '2014-08-04-13-23-01', '2014-08-04-13-15-29',
        '2014-08-04-13-37-12', '2014-08-04-13-37-54', '2014-08-04-13-14-04',
        '2014-08-04-13-14-57', '2014-08-04-13-11-13', '2014-08-04-13-08-01',
        '2014-03-17-13-11-22', '2014-08-04-13-37-57', '2014-08-04-13-22-34',
        '2014-03-17-13-18-30', '2014-08-04-13-04-41', '2014-03-19-14-33-45',
        '2014-08-04-13-08-56', '2014-08-04-13-05-10', '2014-08-04-13-06-53',
        '2014-08-04-13-08-17', '2014-08-04-13-14-08', '2014-05-06-12-17-19',
        '2014-08-04-13-41-10', '2014-08-04-13-22-41', '2014-08-04-13-37-29',
        '2014-08-04-13-16-58', '2014-03-17-13-20-25', '2014-08-04-13-05-06',
        '2014-08-04-13-08-10', '2014-03-17-13-05-15', '2014-08-04-13-11-31',
        '2014-08-04-13-11-53', '2014-08-04-13-13-04', '2014-03-20-10-53-52',
        '2014-08-04-13-21-34', '2014-08-04-13-05-49', '2014-08-04-13-05-22',
        '2014-08-04-13-39-00', '2014-08-04-13-05-45', '2014-03-17-13-06-05',
        '2014-08-04-13-05-42', '2014-08-04-13-15-38', '2014-08-04-13-39-42',
        '2014-06-17-13-46-39', '2014-08-04-13-22-49', '2014-08-04-13-22-02',
        '2014-08-04-13-23-22', '2014-08-04-13-05-19', '2014-08-04-13-09-04',
        '2014-08-04-13-37-16', '2014-08-04-13-39-03', '2014-08-04-13-22-05',
        '2014-08-04-13-11-18', '2014-08-04-13-09-22', '2014-08-04-13-38-56',
        '2014-08-04-13-16-37', '2014-08-04-13-07-54', '2014-08-04-13-37-19',
        '2014-08-04-13-22-53', '2014-05-13-12-01-27', '2014-08-04-13-15-07',
        '2014-08-04-13-22-37', '2014-08-04-13-39-59', '2014-08-04-13-39-50',
        '2014-08-04-13-21-54', '2014-08-04-13-11-01', '2014-08-04-13-23-09',
        '2014-08-04-13-37-41', '2014-08-04-13-13-30', '2014-08-04-13-05-02',
        '2014-08-04-13-14-30', '2014-08-04-13-39-29', '2014-08-04-13-37-45',
        '2014-03-17-13-17-22', '2014-08-04-13-40-04', '2014-03-17-13-03-57',
        '2014-08-04-13-09-27', '2014-08-04-13-06-21', '2014-08-04-13-41-03',
        '2014-08-04-13-06-49', '2014-08-04-13-16-20', '2014-08-04-13-37-22',
        '2014-08-04-13-21-29', '2014-08-04-13-06-31', '2014-08-04-13-16-02',
        '2014-08-04-13-09-13', '2014-03-17-13-14-56', '2014-08-04-13-08-05',
        '2014-05-06-10-50-37', '2014-08-04-13-14-12', '2014-08-04-13-15-02',
        '2014-08-04-13-13-49', '2014-08-04-13-40-07', '2014-08-04-13-23-13',
        '2014-08-04-13-14-53', '2014-08-04-13-08-40', '2014-03-17-13-18-33',
        '2014-08-04-13-39-16', '2014-08-04-13-23-05', '2014-08-04-13-05-26',
        '2014-08-04-13-05-30', '2014-08-04-13-06-12', '2014-08-04-13-05-14',
        '2014-08-04-13-41-18', '2014-03-17-13-15-57', '2014-08-04-13-04-37',
        '2014-08-04-13-14-00', '2014-08-04-13-15-11', '2014-03-17-13-15-42',
        '2014-08-04-13-41-22', '2014-03-17-13-04-03', '2014-08-04-13-11-56',
        '2014-08-04-13-37-49', '2014-08-04-13-14-35', '2014-08-04-13-07-58',
        '2014-08-04-13-06-09', '2014-08-04-13-10-53', '2014-08-04-13-41-14',
        '2014-08-04-13-37-36', '2014-08-04-13-10-57', '2014-08-04-13-13-33',
        '2014-03-17-13-19-59', '2014-08-04-13-13-22', '2014-08-04-13-04-49',
        '2014-08-04-13-13-37', '2014-08-04-13-23-17', '2014-08-04-13-11-40',
        '2014-08-04-13-14-42', '2014-08-04-13-09-00', '2014-08-04-13-13-53',
        '2014-08-04-13-15-49', '2014-03-17-13-13-51', '2014-08-04-13-17-01'
    ],
    'dev': [
        # INVALID AUDIO
        '2015-02-09-13-48-26', '2015-02-09-12-36-46', '2015-01-28-11-49-53',
        '2015-02-04-12-29-49'
    ],
    'test': [
        # INVALID AUDIO
        '2015-02-04-12-36-32', '2015-02-10-13-45-07', '2015-01-27-14-37-33',
        '2015-02-10-14-18-26'
    ]
}


class TudaReader(base.CorpusReader):
    """
    Reader for the TUDA german distant speech corpus (german-speechdata-package-v2.tar.gz).

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
            ids = TudaReader.get_ids_from_folder(sub_path, part)
            utt_ids = []

            for idx in ids:
                add_ids = TudaReader.load_file(sub_path, idx, corpus)
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
            splits[wavtype].append(utt_id)

        for sub_name, sub_utts in splits.items():
            subview_filter = subview.MatchingUtteranceIdxFilter(utterance_idxs=sub_utts)
            subview_corpus = subview.Subview(corpus, filter_criteria=[subview_filter])
            corpus.import_subview('{}{}'.format(prefix, sub_name), subview_corpus)

    @staticmethod
    def get_ids_from_folder(path, part_name):
        """
        Return all ids from the given folder, which have a corresponding beamformedSignal file.
        """
        valid_ids = set({})

        for xml_file in glob.glob(os.path.join(path, '*.xml')):
            idx = os.path.splitext(os.path.basename(xml_file))[0]

            if idx not in BAD_FILES[part_name]:
                valid_ids.add(idx)

        return valid_ids

    @staticmethod
    def load_file(folder_path, idx, corpus):
        """
        Load speaker, file, utterance, labels for the file with the given id.
        """
        xml_path = os.path.join(folder_path, '{}.xml'.format(idx))
        wav_paths = glob.glob(os.path.join(folder_path, '{}_*.wav'.format(idx)))

        if len(wav_paths) == 0:
            return []

        xml_file = open(xml_path, 'r', encoding='utf-8')
        soup = BeautifulSoup(xml_file, 'lxml')

        transcription = soup.recording.cleaned_sentence.string
        transcription_raw = soup.recording.sentence.string
        gender = soup.recording.gender.string
        is_native = soup.recording.muttersprachler.string
        age_class = soup.recording.ageclass.string
        speaker_idx = soup.recording.speaker_id.string

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
