import os

import pytest

from audiomate import corpus
from audiomate import issuers
from audiomate.corpus import io
from tests import resources


@pytest.fixture
def reader():
    return io.TimitReader()


@pytest.fixture
def data_path():
    return resources.sample_corpus_path('timit')


class TestTimitReader:

    def test_load_correct_number_of_tracks(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_tracks == 9

    @pytest.mark.parametrize('idx,path', [
        ('dr1-mkls0-sa1', os.path.join('TRAIN', 'DR1', 'MKLS0', 'SA1.WAV')),
        ('dr1-mkls0-sa2', os.path.join('TRAIN', 'DR1', 'MKLS0', 'SA2.WAV')),
        ('dr1-mrcg0-sx78', os.path.join('TRAIN', 'DR1', 'MRCG0', 'SX78.WAV')),
        ('dr2-mkjo0-si1517', os.path.join('TRAIN', 'DR2', 'MKJO0', 'SI1517.WAV')),
        ('dr2-mrfk0-sx176', os.path.join('TRAIN', 'DR2', 'MRFK0', 'SX176.WAV')),
        ('dr1-fdac1-sa2', os.path.join('TEST', 'DR1', 'FDAC1', 'SA2.WAV')),
        ('dr1-mjsw0-sa1', os.path.join('TEST', 'DR1', 'MJSW0', 'SA1.WAV')),
        ('dr1-mjsw0-sx20', os.path.join('TEST', 'DR1', 'MJSW0', 'SX20.WAV')),
        ('dr2-fpas0-sx224', os.path.join('TEST', 'DR2', 'FPAS0', 'SX224.WAV'))
    ])
    def test_load_tracks(self, idx, path, reader, data_path):
        ds = reader.load(data_path)

        assert ds.tracks[idx].idx == idx
        assert ds.tracks[idx].path == os.path.join(data_path, path)

    def test_load_correct_number_of_utterances(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_utterances == 9

    @pytest.mark.parametrize('idx, issuer_idx', [
        ('dr1-mkls0-sa1', 'KLS0'),
        ('dr1-mkls0-sa2', 'KLS0'),
        ('dr1-mrcg0-sx78', 'RCG0'),
        ('dr2-mkjo0-si1517', 'KJO0'),
        ('dr2-mrfk0-sx176', 'RFK0'),
        ('dr1-fdac1-sa2', 'DAC1'),
        ('dr1-mjsw0-sa1', 'JSW0'),
        ('dr1-mjsw0-sx20', 'JSW0'),
        ('dr2-fpas0-sx224', 'PAS0')
    ])
    def test_load_utterances(self, idx, issuer_idx, reader, data_path):
        ds = reader.load(data_path)

        assert ds.utterances[idx].idx == idx
        assert ds.utterances[idx].track.idx == idx
        assert ds.utterances[idx].issuer.idx == issuer_idx
        assert ds.utterances[idx].start == 0
        assert ds.utterances[idx].end == -1

    def test_load_correct_number_of_issuers(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_issuers == 7

    @pytest.mark.parametrize('idx,gender,num_utt', [
        ('DAC1', issuers.Gender.FEMALE, 1),
        ('JSW0', issuers.Gender.MALE, 2),
        ('PAS0', issuers.Gender.FEMALE, 1),
        ('KLS0', issuers.Gender.MALE, 2),
        ('RCG0', issuers.Gender.MALE, 1),
        ('KJO0', issuers.Gender.MALE, 1),
        ('RFK0', issuers.Gender.MALE, 1)
    ])
    def test_load_issuers(self, idx, gender, num_utt, reader, data_path):
        ds = reader.load(data_path)

        assert ds.issuers[idx].idx == idx
        assert type(ds.issuers[idx]) == issuers.Speaker
        assert ds.issuers[idx].gender == gender
        assert len(ds.issuers[idx].utterances) == num_utt

    @pytest.mark.parametrize('utt_id, text', [
        ('dr1-mkls0-sa1', 'She had your dark suit in greasy wash water all year.'),
        ('dr1-mkls0-sa2', 'Don\'t ask me to carry an oily rag like that.'),
        ('dr1-mrcg0-sx78', 'Doctors prescribe drugs too freely.'),
        ('dr2-mkjo0-si1517', 'Hired, hard lackeys of the warmongering capitalists.'),
        ('dr2-mrfk0-sx176', 'Buying a thoroughbred horse requires intuition and expertise.'),
        ('dr1-fdac1-sa2', 'Don\'t ask me to carry an oily rag like that.'),
        ('dr1-mjsw0-sa1', 'She had your dark suit in greasy wash water all year.'),
        ('dr1-mjsw0-sx20', 'She wore warm, fleecy, woolen overalls.'),
        ('dr2-fpas0-sx224', 'How good is your endurance?')
    ])
    def test_load_raw_transcriptions(self, utt_id, text, reader, data_path):
        ds = reader.load(data_path)

        assert 1 == len(ds.utterances[utt_id].label_lists[corpus.LL_WORD_TRANSCRIPT_RAW])
        assert text == ds.utterances[utt_id].label_lists[corpus.LL_WORD_TRANSCRIPT_RAW][0].value

    @pytest.mark.parametrize('utt_id, num_words, index, word, start, end', [
        ('dr1-mkls0-sa1', 11, 0, 'she', 0.210625, 0.4275),
        ('dr1-mkls0-sa2', 10, 1, 'ask', 0.3625, 0.645),
        ('dr1-mrcg0-sx78', 5, 4, 'freely', 1.8575, 2.2898125),
        ('dr2-mkjo0-si1517', 7, 5, 'warmongering', 1.461125, 2.18275)
    ])
    def test_load_words(self, utt_id, num_words, index, word, start, end, reader, data_path):
        ds = reader.load(data_path)

        assert num_words == len(ds.utterances[utt_id].label_lists[corpus.LL_WORD_TRANSCRIPT])
        assert word == ds.utterances[utt_id].label_lists[corpus.LL_WORD_TRANSCRIPT][index].value
        assert pytest.approx(start) == ds.utterances[utt_id].label_lists[corpus.LL_WORD_TRANSCRIPT][index].start
        assert pytest.approx(end) == ds.utterances[utt_id].label_lists[corpus.LL_WORD_TRANSCRIPT][index].end

    @pytest.mark.parametrize('utt_id, num_phones, index, phone, start, end', [
        ('dr1-mkls0-sa1', 42, 0, 'h#', 0.0, 0.210625),
        ('dr1-mkls0-sa2', 33, 1, 'd', 0.209375, 0.244375),
        ('dr1-mrcg0-sx78', 31, 4, 't', 0.348125, 0.37375),
        ('dr2-mkjo0-si1517', 42, 41, 'h#', 2.9075, 3.1)
    ])
    def test_load_phones(self, utt_id, num_phones, index, phone, start, end, reader, data_path):
        ds = reader.load(data_path)

        assert num_phones == len(ds.utterances[utt_id].label_lists[corpus.LL_PHONE_TRANSCRIPT])
        assert phone == ds.utterances[utt_id].label_lists[corpus.LL_PHONE_TRANSCRIPT][index].value
        assert pytest.approx(start) == ds.utterances[utt_id].label_lists[corpus.LL_PHONE_TRANSCRIPT][index].start
        assert pytest.approx(end) == ds.utterances[utt_id].label_lists[corpus.LL_PHONE_TRANSCRIPT][index].end

    def test_load_correct_number_of_subsets(self, reader, data_path):
        ds = reader.load(data_path)

        assert 2 == ds.num_subviews

    @pytest.mark.parametrize('idx,num_utt', [
        ('TRAIN', 5),
        ('TEST', 4)
    ])
    def test_load_subsets(self, idx, num_utt, reader, data_path):
        ds = reader.load(data_path)

        assert num_utt == ds.subviews[idx].num_utterances
