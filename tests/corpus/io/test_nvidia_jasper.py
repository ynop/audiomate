import json
import os
import pytest

import audiomate
from audiomate.issuers import Speaker, Issuer, Gender
from audiomate.annotations import LabelList, Label
from audiomate.corpus.subset import subview
from audiomate.corpus import io

from tests import resources


def create_sample_dataset(temp_dir):
    ds = audiomate.Corpus(str(temp_dir))

    file_1_path = resources.sample_wav_file('wav_1.wav')
    file_2_path = resources.sample_wav_file('wav_2.wav')
    file_3_path = resources.get_resource_path(['audio_formats', 'flac_1_16k_16b.flac'])

    file_1 = ds.new_file(file_1_path, track_idx='wav_1')
    file_2 = ds.new_file(file_2_path, track_idx='wav_2')
    file_3 = ds.new_file(file_3_path, track_idx='wav_3')

    issuer_1 = Speaker('spk-1', gender=Gender.MALE)
    issuer_2 = Speaker('spk-2', gender=Gender.FEMALE)
    issuer_3 = Issuer('spk-3')

    ds.import_issuers([issuer_1, issuer_2, issuer_3])

    # 2.5951875
    utt_1 = ds.new_utterance('utt-1', file_1.idx, issuer_idx=issuer_1.idx)
    utt_2 = ds.new_utterance('utt-2', file_2.idx, issuer_idx=issuer_2.idx, start=0, end=1.5)
    utt_3 = ds.new_utterance('utt-3', file_2.idx, issuer_idx=issuer_2.idx, start=1.5, end=2.5)
    # 5.0416875
    utt_4 = ds.new_utterance('utt-4', file_3.idx, issuer_idx=issuer_3.idx)

    utt_1.set_label_list(LabelList(audiomate.corpus.LL_WORD_TRANSCRIPT,
                                   labels=[Label('who am i')]))
    utt_2.set_label_list(LabelList(audiomate.corpus.LL_WORD_TRANSCRIPT,
                                   labels=[Label('who are you')]))
    utt_3.set_label_list(LabelList(audiomate.corpus.LL_WORD_TRANSCRIPT,
                                   labels=[Label('who is he')]))
    utt_4.set_label_list(LabelList(audiomate.corpus.LL_WORD_TRANSCRIPT,
                                   labels=[Label('who are they')]))

    train_filter = subview.MatchingUtteranceIdxFilter(utterance_idxs={'utt-1', 'utt-2', 'utt-3'})
    sv_train = subview.Subview(ds, filter_criteria=[train_filter])

    dev_filter = subview.MatchingUtteranceIdxFilter(utterance_idxs={'utt-4'})
    sv_dev = subview.Subview(ds, filter_criteria=[dev_filter])

    ds.import_subview('train', sv_train)
    ds.import_subview('dev', sv_dev)

    return ds


class TestNvidiaJasperWriter:

    def test_default_params(self, tmp_path):
        writer = io.NvidiaJasperWriter()
        ds = create_sample_dataset(tmp_path)

        writer.save(ds, str(tmp_path))

        all_file = tmp_path / 'all.json'
        train_file = tmp_path / 'train.json'
        dev_file = tmp_path / 'dev.json'

        assert all_file.exists()
        assert train_file.exists()
        assert dev_file.exists()

        all_content = json.loads(all_file.read_text())
        assert len(all_content) == 4

        assert all_content[0] == {
            'transcript': 'who is he',
            'files': [{
                'fname': 'audio/utt-3.wav',
                'channels': 1,
                'sample_rate': 16000,
                'duration': pytest.approx(1.0),
                'num_samples': 16000,
                'speed': 1}],
            'original_duration': pytest.approx(1.0),
            'original_num_samples': 16000,
            'utt_idx': 'utt-3'
        }
        assert (tmp_path / all_content[0]['files'][0]['fname']).exists()

        assert all_content[1] == {
            'transcript': 'who are you',
            'files': [{
                'fname': 'audio/utt-2.wav',
                'channels': 1,
                'sample_rate': 16000,
                'duration': pytest.approx(1.5),
                'num_samples': 24000,
                'speed': 1
            }],
            'original_duration': pytest.approx(1.5),
            'original_num_samples': 24000,
            'utt_idx': 'utt-2'
        }
        assert (tmp_path / all_content[1]['files'][0]['fname']).exists()

        assert all_content[2] == {
            'transcript': 'who am i',
            'files': [{
                'fname': os.path.relpath(ds.tracks['wav_1'].path, str(tmp_path)),
                'channels': 1,
                'sample_rate': 16000,
                'duration': pytest.approx(2.5951875),
                'num_samples': 41523,
                'speed': 1
            }],
            'original_duration': pytest.approx(2.5951875),
            'original_num_samples': 41523,
            'utt_idx': 'utt-1'
        }
        abspath = os.path.abspath(os.path.join(str(tmp_path), all_content[2]['files'][0]['fname']))
        assert os.path.isfile(abspath)

        # Use large margin for duration/num_samples since some backends return different results
        assert all_content[3] == {
            'transcript': 'who are they',
            'files': [{
                'fname': 'audio/utt-4.wav',
                'channels': 1,
                'sample_rate': 16000,
                'duration': pytest.approx(6.464, abs=1e-1),
                'num_samples': pytest.approx(103424, abs=1600),
                'speed': 1
            }],
            'original_duration': pytest.approx(6.464, abs=1e-1),
            'original_num_samples': pytest.approx(103424, abs=1600),
            'utt_idx': 'utt-4'
        }
        assert (tmp_path / all_content[3]['files'][0]['fname']).exists()

        train_content = json.loads(train_file.read_text())
        assert len(train_content) == 3

        dev_content = json.loads(dev_file.read_text())
        assert len(dev_content) == 1

    def test_export_all(self, tmp_path):
        writer = io.NvidiaJasperWriter(export_all_audio=True)
        ds = create_sample_dataset(tmp_path)

        writer.save(ds, str(tmp_path))

        all_file = tmp_path / 'all.json'
        train_file = tmp_path / 'train.json'
        dev_file = tmp_path / 'dev.json'

        assert all_file.exists()
        assert train_file.exists()
        assert dev_file.exists()

        all_content = json.loads(all_file.read_text())
        assert len(all_content) == 4

        assert all_content[2] == {
            'transcript': 'who am i',
            'files': [{
                'fname': 'audio/utt-1.wav',
                'channels': 1,
                'sample_rate': 16000,
                'duration': pytest.approx(2.5951875),
                'num_samples': 41523,
                'speed': 1
            }],
            'original_duration': pytest.approx(2.5951875),
            'original_num_samples': 41523,
            'utt_idx': 'utt-1'
        }
        assert (tmp_path / all_content[2]['files'][0]['fname']).exists()

        train_content = json.loads(train_file.read_text())
        assert len(train_content) == 3

        dev_content = json.loads(dev_file.read_text())
        assert len(dev_content) == 1
