import os
import struct

import numpy as np
import scipy

import audiomate
from audiomate import tracks
from audiomate import annotations
from audiomate import issuers
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
          Describes how a data set has to be structured to be understood
          by Kaldi and the format of the individual files.
    """

    def __init__(self, main_label_list_idx=audiomate.corpus.LL_WORD_TRANSCRIPT,
                 main_feature_idx='default', include_invalid_items=False):
        super(KaldiReader, self).__init__(include_invalid_items=include_invalid_items)
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
                    gender = issuers.Gender.MALE
                else:
                    gender = issuers.Gender.FEMALE

                speaker = issuers.Speaker(speaker_idx, gender=gender)
                corpus.import_issuers(speaker)

    @staticmethod
    def read_utterances(segments_path, corpus, utt2spk):
        # load utterances
        if os.path.isfile(segments_path):
            utterances = textfile.read_separated_lines_with_first_key(
                segments_path,
                separator=' ',
                max_columns=4
            )

            for utt_id, utt_info in utterances.items():
                start = 0
                end = float('inf')

                if len(utt_info) > 1:
                    start = float(utt_info[1])

                if len(utt_info) > 2:
                    end = float(utt_info[2])

                    if end == -1:
                        end = float('inf')

                speaker_idx = None

                if utt_id in utt2spk.keys():
                    speaker_idx = utt2spk[utt_id].idx

                corpus.new_utterance(
                    utt_id, utt_info[0],
                    issuer_idx=speaker_idx,
                    start=start,
                    end=end
                )
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
            ll = annotations.LabelList.create_single(
                transcription,
                idx=audiomate.corpus.LL_WORD_TRANSCRIPT
            )
            corpus.utterances[utt_id].set_label_list(ll)


class KaldiWriter(base.CorpusWriter):
    """
    Supports writing data sets in Kaldi format.

    Args:
        main_label_list_idx (str): The idx of the label-list to use
                                   for writing to transcriptions file.
        main_feature_idx (str): The idx of the feature-container to export.
        use_utt_idx_if_no_speaker_available (bool): If ``True``, the
                                                    utterance-idx is used as
                                                    speaker-idx in the utt2spk
                                                    file, if no speaker exists
                                                    for an utterance.
        create_spk2gender (bool): If ``True`` creates the file spk2gender.
        default_gender (str): If ``create_spk2gender==True`` and the gender of
                              an issuer is not known,
                              this default value will be used (default 'm').
        prefix_utterances_with_speaker (bool): If ``True``, add a prefix in
                                               form of the issuer-idx to
                                               every utterance.
        use_absolute_times (bool): If ``True``, doesn't use -1 for segment ends,
                                   but reads the audio to get absolute duration.

    .. seealso::

       `Kaldi: Data preparation <http://kaldi-asr.org/doc/data_prep.html>`_
          Describes how a data set has to be structured to be
          understood by Kaldi and the format of the individual files.
    """

    def __init__(self, main_label_list_idx=audiomate.corpus.LL_WORD_TRANSCRIPT,
                 main_feature_idx='default', use_utt_idx_if_no_speaker_available=True,
                 create_spk2gender=False, default_gender='m',
                 prefix_utterances_with_speaker=True, use_absolute_times=False):
        self.main_label_list_idx = main_label_list_idx
        self.main_feature_idx = main_feature_idx
        self.use_utt_idx_if_no_speaker_available = use_utt_idx_if_no_speaker_available
        self.create_spk2gender = create_spk2gender
        self.default_gender = default_gender
        self.prefix_utterances_with_speaker = prefix_utterances_with_speaker
        self.use_absolute_times = use_absolute_times

    @classmethod
    def type(cls):
        return 'kaldi'

    def _save(self, corpus, path):
        wav_file_path = os.path.join(path, WAV_FILE_NAME)
        spk2gender_path = os.path.join(path, SPK2GENDER_FILE_NAME)
        utt2spk_path = os.path.join(path, UTT2SPK_FILE_NAME)
        segments_path = os.path.join(path, SEGMENTS_FILE_NAME)
        text_path = os.path.join(path, TRANSCRIPTION_FILE_NAME)

        KaldiWriter.write_tracks(wav_file_path, corpus, path)
        self._write_segments(segments_path, corpus)
        self._write_utt_to_issuer_mapping(utt2spk_path, corpus)
        self._write_transcriptions(text_path, corpus)
        self._write_features(path, corpus)

        if self.create_spk2gender:
            self._write_genders(spk2gender_path, corpus)

    @staticmethod
    def write_tracks(file_path, corpus, path):
        file_records = []

        export_path = os.path.join(path, 'audio')

        for track in corpus.tracks.values():
            if isinstance(track, tracks.FileTrack):
                file_records.append([
                    track.idx,
                    KaldiWriter.extended_filename(track)
                ])

            elif isinstance(track, tracks.ContainerTrack):
                if not os.path.isdir(export_path):
                    os.makedirs(export_path)

                target_path = os.path.join(
                    export_path,
                    '{}.wav'.format(track.idx)
                )

                max_value = np.iinfo(np.int16).max
                samples = (track.read_samples() * max_value).astype(np.int16)
                sampling_rate = track.sampling_rate
                scipy.io.wavfile.write(target_path, sampling_rate, samples)

                file_records.append([
                    track.idx,
                    target_path
                ])

        textfile.write_separated_lines(
            file_path,
            file_records,
            separator=' ',
            sort_by_column=0
        )

    @staticmethod
    def extended_filename(file_track):
        """
        Create extended filename.
        Kaldi only supports wav.
        Therefore other files have to be converted using sox.
        """
        ext = os.path.splitext(file_track.path)[1]
        abs_path = os.path.abspath(file_track.path)

        if ext == '.wav':
            return abs_path
        else:
            return 'sox {} -t wav - |'.format(abs_path)

    def _write_segments(self, utterance_path, corpus):
        utterances = corpus.utterances.values()
        utterance_records = {}

        for u in utterances:
            utt_idx = self._get_utt_idx(u)
            track_idx = u.track.idx
            start = u.start
            end = u.end

            if end == float('inf'):
                if self.use_absolute_times:
                    end = u.end_abs
                else:
                    end = -1

            utterance_records[utt_idx] = [track_idx, start, end]

        textfile.write_separated_lines(
            utterance_path,
            utterance_records,
            separator=' ',
            sort_by_column=0
        )

    def _write_genders(self, gender_path, corpus):
        genders = {}

        for issuer in corpus.issuers.values():
            gender = self.default_gender

            if type(issuer) is issuers.Speaker:
                if issuer.gender == issuers.Gender.MALE:
                    gender = 'm'
                elif issuer.gender == issuers.Gender.FEMALE:
                    gender = 'f'

            genders[issuer.idx] = gender

        if len(genders) > 0:
            textfile.write_separated_lines(
                gender_path,
                genders,
                separator=' ',
                sort_by_column=0
            )

    def _write_transcriptions(self, text_path, corpus):
        transcriptions = {}

        for utterance in corpus.utterances.values():
            utt_idx = self._get_utt_idx(utterance)

            if self.main_label_list_idx in utterance.label_lists.keys():
                label_list = utterance.label_lists[self.main_label_list_idx]
                transcriptions[utt_idx] = ' '.join(l.value for l in label_list)

        textfile.write_separated_lines(
            text_path,
            transcriptions,
            separator=' ',
            sort_by_column=0
        )

    def _write_utt_to_issuer_mapping(self, utt_issuer_path, corpus):
        utt_issuer_records = {}

        for utterance in corpus.utterances.values():
            utt_idx = self._get_utt_idx(utterance)
            if utterance.issuer is not None:
                utt_issuer_records[utt_idx] = utterance.issuer.idx
            elif self.use_utt_idx_if_no_speaker_available:
                utt_issuer_records[utt_idx] = utt_idx

        textfile.write_separated_lines(
            utt_issuer_path,
            utt_issuer_records,
            separator=' ',
            sort_by_column=0
        )

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

            ark_path = os.path.join(path, '{}.ark'.format(FEATS_FILE_NAME))
            ark_path = os.path.abspath(ark_path)
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

            # check if  it is a binary ark
            binary = f.read(2)

            if binary != b'\x00B':
                msg = 'The ark "{}" is not in binary format!'
                raise ValueError(msg.format(rx_specifier))

            # check if data type is float 32
            archive_format = f.read(3)
            if archive_format != b'FM ':
                msg = 'The ark "{}" has not float 32 type!'
                raise ValueError(msg.format(rx_specifier))

            # get number of mfcc features
            f.read(1)
            num_frames = struct.unpack('<i', f.read(4))[0]

            # get size of mfcc features
            f.read(1)
            feature_size = struct.unpack('<i', f.read(4))[0]

            # read feature data
            data = f.read(num_frames * feature_size * sample_format)

            feature_vector = np.frombuffer(data, dtype='float32')
            feature_matrix = np.reshape(
                feature_vector,
                (num_frames, feature_size)
            )

            return feature_matrix

    def _get_utt_idx(self, utt):
        if (self.prefix_utterances_with_speaker and
                utt.issuer is not None and
                not utt.idx.startswith(utt.issuer.idx)):
            return '{}-{}'.format(utt.issuer.idx, utt.idx)
        else:
            return utt.idx

    @staticmethod
    def write_float_matrices(scp_path, ark_path, matrices):
        """
        Write the given dict matrices (utt-id/float ndarray)
        to the given scp and ark files.
        """

        scp_entries = []

        with open(ark_path, 'wb') as f:
            for utterance_id in sorted(list(matrices.keys())):
                matrix = matrices[utterance_id]

                if matrix.dtype != np.float32:
                    msg = 'Features of utterance "{}" are have not type float 32!'
                    raise ValueError(msg.format(utterance_id))

                f.write(('{} '.format(utterance_id)).encode('utf-8'))

                offset = f.tell()

                f.write(b'\x00B')
                f.write(b'FM ')
                f.write(b'\x04')
                f.write(struct.pack('<i', np.size(matrix, 0)))
                f.write(b'\x04')
                f.write(struct.pack('<i', np.size(matrix, 1)))

                flattened = matrix.reshape(
                    np.size(matrix, 0) * np.size(matrix, 1)
                )
                flattened.tofile(f, sep='')

                scp_entries.append('{} {}:{}'.format(
                    utterance_id,
                    ark_path,
                    offset
                ))

        with open(scp_path, 'w') as f:
            f.write('\n'.join(scp_entries))
