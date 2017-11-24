import os
import struct

import numpy as np

from . import base
from pingu.corpus import assets
from pingu.utils import textfile

WAV_FILE_NAME = 'wav.scp'
SEGMENTS_FILE_NAME = 'segments'
UTT2SPK_FILE_NAME = 'utt2spk'
SPK2GENDER_FILE_NAME = 'spk2gender'
TRANSCRIPTION_FILE_NAME = 'text'
FEATS_FILE_NAME = 'feats'


class KaldiLoader(base.CorpusLoader):

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

        return missing_files or None

    def _load(self, corpus):
        # load wavs
        wav_file_path = os.path.join(corpus.path, WAV_FILE_NAME)
        for file_idx, file_path in textfile.read_key_value_lines(wav_file_path, separator=' ').items():
            corpus.add_file(file_path, file_idx=file_idx)

        # load utterances
        utt2spk_path = os.path.join(corpus.path, UTT2SPK_FILE_NAME)
        utt2spk = {}

        if os.path.isfile(utt2spk_path):
            utt2spk = textfile.read_key_value_lines(utt2spk_path, separator=' ')

        segments_path = os.path.join(corpus.path, SEGMENTS_FILE_NAME)

        if os.path.isfile(segments_path):
            for utt_id, utt_info in textfile.read_separated_lines_with_first_key(segments_path, separator=' ', max_columns=4).items():
                start = None
                end = None

                if len(utt_info) > 1:
                    start = utt_info[1]

                if len(utt_info) > 2:
                    end = utt_info[2]

                speaker_idx = None

                if utt_id in utt2spk.keys():
                    speaker_idx = utt2spk[utt_id]
                    if speaker_idx not in corpus.speakers.keys():
                        corpus.add_speaker(speaker_idx=speaker_idx)

                corpus.add_utterance(utt_info[0], utterance_idx=utt_id, speaker_idx=speaker_idx, start=start, end=end)
        else:
            for file_idx in corpus.files.keys():
                speaker_idx = None

                if file_idx in utt2spk.keys():
                    speaker_idx = utt2spk[file_idx]
                    if speaker_idx not in corpus.speakers.keys():
                        corpus.add_speaker(speaker_idx=speaker_idx)

                corpus.add_utterance(file_idx, utterance_idx=file_idx, speaker_idx=speaker_idx)

        # load transcriptions
        text_path = os.path.join(corpus.path, TRANSCRIPTION_FILE_NAME)
        for utt_id, transcription in textfile.read_key_value_lines(text_path, separator=' ').items():
            corpus.add_segmentation(utt_id, segments=transcription)

    def _save(self, corpus, path):
        kaldi_files = {f.idx: os.path.abspath(os.path.join(path, f.path)) for f in corpus.files.values()}

        # Write files
        file_path = os.path.join(path, WAV_FILE_NAME)
        textfile.write_separated_lines(file_path, kaldi_files, separator=' ', sort_by_column=0)

        # Write utterances
        utterance_path = os.path.join(path, SEGMENTS_FILE_NAME)
        utterance_records = {utterance.idx: [utterance.file_idx, utterance.start, utterance.end] for utterance in corpus.utterances.values()}
        textfile.write_separated_lines(utterance_path, utterance_records, separator=' ', sort_by_column=0)

        # Write utt2spk
        utt2spk_path = os.path.join(path, UTT2SPK_FILE_NAME)
        utt2spk_records = {utterance.idx: utterance.speaker_idx for utterance in corpus.utterances.values()}
        textfile.write_separated_lines(utt2spk_path, utt2spk_records, separator=' ', sort_by_column=0)

        # Write speakers
        gender_path = os.path.join(path, SPK2GENDER_FILE_NAME)
        speaker_data = {spk.idx: spk.gender.value for spk in corpus.speakers.values()}
        textfile.write_separated_lines(gender_path, speaker_data, separator=' ', sort_by_column=0)

        # Write segmentations
        transcriptions = {}

        for utterance_idx, utt_segmentations in corpus.segmentations.items():
            if data.Segmentation.TEXT_SEGMENTATION in utt_segmentations.keys():
                transcriptions[utterance_idx] = utt_segmentations[data.Segmentation.TEXT_SEGMENTATION].to_text()

        text_path = os.path.join(path, TRANSCRIPTION_FILE_NAME)
        textfile.write_separated_lines(text_path, transcriptions, separator=' ', sort_by_column=0)

        # Write features
        if self.main_features is not None:
            fc = corpus.features[self.main_features]
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
            yield utterance_id, KaldiLoader.read_float_matrix(rx_specifier)

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
                flattened.tofile(f, sep="")

                scp_entries.append('{} {}:{}'.format(utterance_id, ark_path, offset))

        with open(scp_path, 'w') as f:
            f.write('\n'.join(scp_entries))
