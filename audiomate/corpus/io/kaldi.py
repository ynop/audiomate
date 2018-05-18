import os
import struct

import numpy as np

import audiomate
from audiomate.corpus import assets
from audiomate.utils import textfile
from . import base
from . import default

WAV_FILE_NAME = 'wav.scp'
SEGMENTS_FILE_NAME = 'segments'
UTT2SPK_FILE_NAME = 'utt2spk'
SPK2GENDER_FILE_NAME = 'spk2gender'
TRANSCRIPTION_FILE_NAME = 'text'
FEATS_FILE_NAME = 'feats'


class KaldiReader(base.CorpusReader):
    """
    Supports reading data sets in Kaldi format.

    .. seealso::

       `Kaldi: Data preparation <http://kaldi-asr.org/doc/data_prep.html>`_
          Describes how a data set has to be structured to be understood by Kaldi and the format of the individual
          files.
    """

    def __init__(self, main_label_list_idx='default', main_feature_idx='default'):
        self.main_label_list_idx = main_label_list_idx
        self.main_feature_idx = main_feature_idx

    @classmethod
    def type(cls):
        return 'kaldi'

    def _check_for_missing_files(self, path):
        necessary_files = [WAV_FILE_NAME, TRANSCRIPTION_FILE_NAME]
        missing_files = []

        for file_name in necessary_files:
            file_path = os.path.join(path, file_name)

            if not os.path.isfile(file_path):
                missing_files.append(file_name)

        return missing_files

    def _load(self, path):
        wav_file_path = os.path.join(path, WAV_FILE_NAME)
        spk2gender_path = os.path.join(path, SPK2GENDER_FILE_NAME)
        utt2spk_path = os.path.join(path, UTT2SPK_FILE_NAME)
        segments_path = os.path.join(path, SEGMENTS_FILE_NAME)
        text_path = os.path.join(path, TRANSCRIPTION_FILE_NAME)

        corpus = audiomate.Corpus(path=path)

        default.DefaultReader.read_files(wav_file_path, corpus)
        KaldiReader.read_genders(spk2gender_path, corpus)
        utt2spk = default.DefaultReader.read_utt_to_issuer_mapping(utt2spk_path, corpus)
        KaldiReader.read_utterances(segments_path, corpus, utt2spk)
        KaldiReader.read_transcriptions(text_path, corpus)

        return corpus

    @staticmethod
    def read_genders(genders_path, corpus):
        if os.path.isfile(genders_path):
            speakers = textfile.read_key_value_lines(genders_path, separator=' ')

            for speaker_idx, gender_str in speakers.items():
                if gender_str == 'm':
                    gender = assets.Gender.MALE
                else:
                    gender = assets.Gender.FEMALE

                speaker = assets.Speaker(speaker_idx, gender=gender)
                corpus.import_issuers(speaker)

    @staticmethod
    def read_utterances(segments_path, corpus, utt2spk):
        # load utterances
        if os.path.isfile(segments_path):
            utterances = textfile.read_separated_lines_with_first_key(segments_path, separator=' ',
                                                                      max_columns=4)
            for utt_id, utt_info in utterances.items():
                start = 0
                end = -1

                if len(utt_info) > 1:
                    start = float(utt_info[1])

                if len(utt_info) > 2:
                    end = float(utt_info[2])

                speaker_idx = None

                if utt_id in utt2spk.keys():
                    speaker_idx = utt2spk[utt_id].idx

                corpus.new_utterance(utt_id, utt_info[0], issuer_idx=speaker_idx, start=start,
                                     end=end)
        else:
            for file_idx in corpus.files.keys():
                speaker_idx = None

                if file_idx in utt2spk.keys():
                    speaker_idx = utt2spk[file_idx].idx

                corpus.new_utterance(file_idx, file_idx, issuer_idx=speaker_idx)

    @staticmethod
    def read_transcriptions(text_path, corpus):
        transcriptions = textfile.read_key_value_lines(text_path, separator=' ')
        for utt_id, transcription in transcriptions.items():
            ll = assets.LabelList(labels=[assets.Label(transcription)])
            corpus.utterances[utt_id].set_label_list(ll)


class KaldiWriter(base.CorpusWriter):
    """
    Supports writing data sets in Kaldi format.

    .. seealso::

       `Kaldi: Data preparation <http://kaldi-asr.org/doc/data_prep.html>`_
          Describes how a data set has to be structured to be understood by Kaldi and the format of the individual
          files.
    """

    def __init__(self, main_label_list_idx='default', main_feature_idx='default'):
        self.main_label_list_idx = main_label_list_idx
        self.main_feature_idx = main_feature_idx

    @classmethod
    def type(cls):
        return 'kaldi'

    def _save(self, corpus, path):
        wav_file_path = os.path.join(path, WAV_FILE_NAME)
        spk2gender_path = os.path.join(path, SPK2GENDER_FILE_NAME)
        utt2spk_path = os.path.join(path, UTT2SPK_FILE_NAME)
        segments_path = os.path.join(path, SEGMENTS_FILE_NAME)
        text_path = os.path.join(path, TRANSCRIPTION_FILE_NAME)

        default.DefaultWriter.write_files(wav_file_path, corpus, path)
        default.DefaultWriter.write_utterances(segments_path, corpus)
        default.DefaultWriter.write_utt_to_issuer_mapping(utt2spk_path, corpus)

        self._write_genders(spk2gender_path, corpus)
        self._write_transcriptions(text_path, corpus)
        self._write_features(path, corpus)

    def _write_genders(self, gender_path, corpus):
        genders = {}

        for issuer in corpus.issuers.values():
            if type(issuer) == assets.Speaker:
                if issuer.gender == assets.Gender.MALE:
                    genders[issuer.idx] = 'm'
                elif issuer.gender == assets.Gender.FEMALE:
                    genders[issuer.idx] = 'f'

        if len(genders) > 0:
            textfile.write_separated_lines(gender_path, genders, separator=' ', sort_by_column=0)

    def _write_transcriptions(self, text_path, corpus):
        transcriptions = {}

        for utterance in corpus.utterances.values():
            if self.main_label_list_idx in utterance.label_lists.keys():
                label_list = utterance.label_lists[self.main_label_list_idx]
                transcriptions[utterance.idx] = ' '.join([l.value for l in label_list])

        textfile.write_separated_lines(text_path, transcriptions, separator=' ', sort_by_column=0)

    def _write_features(self, path, corpus):
        if self.main_feature_idx in corpus.feature_containers.keys():
            fc = corpus.features_containers[self.main_feature_idx]
            fc.open()
            matrices = {}

            for utt_id in corpus.utterances.keys():
                matrix = fc.get(utt_id)

                if matrix is not None:
                    matrices[utt_id] = matrix

            fc.close()

            ark_path = os.path.abspath(os.path.join(path, '{}.ark'.format(FEATS_FILE_NAME)))
            scp_path = os.path.join(path, '{}.scp'.format(FEATS_FILE_NAME))

            self.write_float_matrices(scp_path, ark_path, matrices)

    @staticmethod
    def feature_scp_generator(path):
        """ Return a generator over all feature matrices defined in a scp. """

        scp_entries = textfile.read_key_value_lines(path, separator=' ')

        for utterance_id, rx_specifier in scp_entries.items():
            yield utterance_id, KaldiWriter.read_float_matrix(rx_specifier)

    @staticmethod
    def read_float_matrix(rx_specifier):
        """ Return float matrix as np array for the given rx specifier. """

        path, offset = rx_specifier.strip().split(':', maxsplit=1)
        offset = int(offset)
        sample_format = 4

        with open(path, 'rb') as f:
            # move to offset
            f.seek(offset)

            # assert binary ark
            binary = f.read(2)
            assert (binary == b'\x00B')

            # assert type float 32
            format = f.read(3)
            assert (format == b'FM ')

            # get number of mfcc features
            f.read(1)
            num_frames = struct.unpack('<i', f.read(4))[0]

            # get size of mfcc features
            f.read(1)
            feature_size = struct.unpack('<i', f.read(4))[0]

            # read feature data
            data = f.read(num_frames * feature_size * sample_format)

            feature_vector = np.frombuffer(data, dtype='float32')
            feature_matrix = np.reshape(feature_vector, (num_frames, feature_size))

            return feature_matrix

    @staticmethod
    def write_float_matrices(scp_path, ark_path, matrices):
        """ Write the given dict matrices (utt-id/float ndarray) to the given scp and ark files. """

        scp_entries = []

        with open(ark_path, 'wb') as f:
            for utterance_id in sorted(list(matrices.keys())):
                matrix = matrices[utterance_id]

                assert (matrix.dtype == np.float32)

                f.write(('{} '.format(utterance_id)).encode('utf-8'))

                offset = f.tell()

                f.write(b'\x00B')
                f.write(b'FM ')
                f.write(b'\x04')
                f.write(struct.pack('<i', np.size(matrix, 0)))
                f.write(b'\x04')
                f.write(struct.pack('<i', np.size(matrix, 1)))

                flattened = matrix.reshape(np.size(matrix, 0) * np.size(matrix, 1))
                flattened.tofile(f, sep='')

                scp_entries.append('{} {}:{}'.format(utterance_id, ark_path, offset))

        with open(scp_path, 'w') as f:
            f.write('\n'.join(scp_entries))
