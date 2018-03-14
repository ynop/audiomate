import os
import glob

from bs4 import BeautifulSoup

import pingu
from pingu.corpus import assets
from pingu.corpus.subset import subview
from . import base

SUBSETS = ['train', 'dev', 'test']
WAV_SUFFIX = '-beamformedSignal'

BAD_FILES = {
    'train': [
        '2014-08-05-11-08-34-Parliament',
        '2014-03-24-13-39-24',
        '2014-03-18-15-29-23',
        '2014-03-18-15-28-52',
        '2014-03-27-11-50-33'
    ],
    'dev': [

        '2015-02-09-13-48-26',
        '2015-02-04-12-29-49',
        '2015-01-28-11-49-53',
        '2015-02-09-12-36-46'
    ],
    'test': [
        '2015-02-10-13-45-07',
        '2015-02-10-14-18-26',
        '2015-01-27-14-37-33',
        '2015-02-04-12-36-32'
    ]
}


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
        corpus = pingu.Corpus(path=path)

        for part in SUBSETS:
            sub_path = os.path.join(path, part)
            ids = TudaReader.get_ids_from_folder(sub_path, part)

            for idx in ids:
                TudaReader.load_file(sub_path, idx, corpus)

            subview_filter = subview.MatchingUtteranceIdxFilter(utterance_idxs=ids)
            subview_corpus = subview.Subview(corpus, filter_criteria=[subview_filter])
            corpus.import_subview(part, subview_corpus)

        return corpus

    @staticmethod
    def get_ids_from_folder(path, part_name):
        """
        Return all ids from the given folder, which have a corresponding beamformedSignal file.
        """
        valid_ids = set({})

        for xml_file in glob.glob(os.path.join(path, '*.xml')):
            idx = os.path.splitext(os.path.basename(xml_file))[0]

            if idx not in BAD_FILES[part_name]:
                beamformed_path = os.path.join(path, '{}{}.wav'.format(idx, WAV_SUFFIX))

                if os.path.isfile(beamformed_path):
                    valid_ids.add(idx)

        return valid_ids

    @staticmethod
    def load_file(folder_path, idx, corpus):
        """
        Load speaker, file, utterance, labels for the file with the given id.
        """
        xml_path = os.path.join(folder_path, '{}.xml'.format(idx))
        wav_path = os.path.join(folder_path, '{}{}.wav'.format(idx, WAV_SUFFIX))

        xml_file = open(xml_path, 'r', encoding='utf-8')
        soup = BeautifulSoup(xml_file, 'lxml')

        transcription = soup.recording.cleaned_sentence.string
        transcription_raw = soup.recording.sentence.string
        gender = soup.recording.gender.string
        speaker_idx = soup.recording.speaker_id.string

        if speaker_idx not in corpus.issuers.keys():
            corpus.new_issuer(speaker_idx, info={'gender': gender})

        corpus.new_file(wav_path, idx)
        utt = corpus.new_utterance(idx, idx, speaker_idx)
        utt.set_label_list(assets.LabelList('transcription', labels=[
            assets.Label(transcription)
        ]))
        utt.set_label_list(assets.LabelList('transcription_raw', labels=[
            assets.Label(transcription_raw)
        ]))
