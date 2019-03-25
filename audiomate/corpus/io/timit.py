import os
import glob

import audiomate
from audiomate import annotations
from audiomate import issuers
from audiomate.corpus import subset
from . import base
from audiomate.utils import textfile


class TimitReader(base.CorpusReader):
    """
    Reader for the TIMIT Corpus.

    .. seealso::

       `TIMIT <https://github.com/philipperemy/timit>`_
          Download page
    """

    @classmethod
    def type(cls):
        return 'timit'

    def _check_for_missing_files(self, path):
        return []

    def _load(self, path):
        corpus = audiomate.Corpus(path=path)

        for part in ['TEST', 'TRAIN']:
            part_path = os.path.join(path, part)
            part_utt_ids = set()

            for region in os.listdir(part_path):
                region_path = os.path.join(part_path, region)

                if os.path.isdir(region_path):

                    for speaker_abbr in os.listdir(region_path):
                        speaker_path = os.path.join(region_path, speaker_abbr)
                        speaker_idx = speaker_abbr[1:]

                        if speaker_idx not in corpus.issuers.keys():
                            issuer = issuers.Speaker(speaker_idx)

                            if speaker_abbr[:1] == 'M':
                                issuer.gender = issuers.Gender.MALE
                            elif speaker_abbr[:1] == 'F':
                                issuer.gender = issuers.Gender.FEMALE

                            corpus.import_issuers(issuer)

                        for wav_path in glob.glob(os.path.join(speaker_path, '*.WAV')):
                            sentence_idx = os.path.splitext(os.path.basename(wav_path))[0]
                            utt_idx = '{}-{}-{}'.format(region, speaker_abbr, sentence_idx).lower()
                            part_utt_ids.add(utt_idx)

                            raw_text_path = os.path.join(speaker_path, '{}.TXT'.format(sentence_idx))
                            raw_text = textfile.read_separated_lines(raw_text_path, separator=' ', max_columns=3)[0][2]

                            words_path = os.path.join(speaker_path, '{}.WRD'.format(sentence_idx))
                            words = textfile.read_separated_lines(words_path, separator=' ', max_columns=3)

                            phones_path = os.path.join(speaker_path, '{}.PHN'.format(sentence_idx))
                            phones = textfile.read_separated_lines(phones_path, separator=' ', max_columns=3)

                            corpus.new_file(wav_path, utt_idx)
                            utt = corpus.new_utterance(utt_idx, utt_idx, speaker_idx)

                            raw_ll = annotations.LabelList.create_single(raw_text,
                                                                         idx=audiomate.corpus.LL_WORD_TRANSCRIPT_RAW)
                            utt.set_label_list(raw_ll)

                            word_ll = annotations.LabelList(idx=audiomate.corpus.LL_WORD_TRANSCRIPT)

                            for record in words:
                                start = int(record[0]) / 16000
                                end = int(record[1]) / 16000
                                word_ll.addl(record[2], start=start, end=end)

                            utt.set_label_list(word_ll)

                            phone_ll = annotations.LabelList(idx=audiomate.corpus.LL_PHONE_TRANSCRIPT)

                            for record in phones:
                                start = int(record[0]) / 16000
                                end = int(record[1]) / 16000
                                phone_ll.addl(record[2], start=start, end=end)

                            utt.set_label_list(phone_ll)

            filter = subset.MatchingUtteranceIdxFilter(utterance_idxs=part_utt_ids)
            subview = subset.Subview(corpus, filter_criteria=[filter])
            corpus.import_subview(part, subview)

        return corpus
