import os

import pytest

from audiomate.corpus import io
from audiomate.corpus import assets
from tests import resources


@pytest.fixture
def reader():
    return io.TimitReader()


@pytest.fixture
def data_path():
    return resources.sample_corpus_path('timit')


class TestTimitReader:

    def test_load_correct_number_of_files(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_files == 9

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
    def test_load_files(self, idx, path, reader, data_path):
        ds = reader.load(data_path)

        assert ds.files[idx].idx == idx
        assert ds.files[idx].path == os.path.join(data_path, path)

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
        assert ds.utterances[idx].file.idx == idx
        assert ds.utterances[idx].issuer.idx == issuer_idx
        assert ds.utterances[idx].start == 0
        assert ds.utterances[idx].end == -1

    def test_load_correct_number_of_issuers(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_issuers == 7

    @pytest.mark.parametrize('idx,gender,num_utt', [
        ('DAC1', assets.Gender.FEMALE, 1),
        ('JSW0', assets.Gender.MALE, 2),
        ('PAS0', assets.Gender.FEMALE, 1),
        ('KLS0', assets.Gender.MALE, 2),
        ('RCG0', assets.Gender.MALE, 1),
        ('KJO0', assets.Gender.MALE, 1),
        ('RFK0', assets.Gender.MALE, 1)
    ])
    def test_load_issuers(self, idx, gender, num_utt, reader, data_path):
        ds = reader.load(data_path)

        assert ds.issuers[idx].idx == idx
        assert type(ds.issuers[idx]) == assets.Speaker
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

        assert 1 == len(ds.utterances[utt_id].label_lists['raw_transcription'])
        assert text == ds.utterances[utt_id].label_lists['raw_transcription'][0].value

    @pytest.mark.parametrize('utt_id, num_words, index, word, start, end', [
        ('dr1-mkls0-sa1', 11, 0, 'she', 0.210625, 0.4275),
        ('dr1-mkls0-sa2', 10, 1, 'ask', 0.3625, 0.645),
        ('dr1-mrcg0-sx78', 5, 4, 'freely', 1.8575, 2.2898125),
        ('dr2-mkjo0-si1517', 7, 5, 'warmongering', 1.461125, 2.18275)
    ])
    def test_load_words(self, utt_id, num_words, index, word, start, end, reader, data_path):
        ds = reader.load(data_path)

        assert num_words == len(ds.utterances[utt_id].label_lists['words'])
        assert word == ds.utterances[utt_id].label_lists['words'][index].value
        assert pytest.approx(start) == ds.utterances[utt_id].label_lists['words'][index].start
        assert pytest.approx(end) == ds.utterances[utt_id].label_lists['words'][index].end

    @pytest.mark.parametrize('utt_id, num_phones, index, phone, start, end', [
        ('dr1-mkls0-sa1', 42, 0, 'h#', 0.0, 0.210625),
        ('dr1-mkls0-sa2', 33, 1, 'd', 0.209375, 0.244375),
        ('dr1-mrcg0-sx78', 31, 4, 't', 0.348125, 0.37375),
        ('dr2-mkjo0-si1517', 42, 41, 'h#', 2.9075, 3.1)
    ])
    def test_load_phones(self, utt_id, num_phones, index, phone, start, end, reader, data_path):
        ds = reader.load(data_path)

        assert num_phones == len(ds.utterances[utt_id].label_lists['phones'])
        assert phone == ds.utterances[utt_id].label_lists['phones'][index].value
        assert pytest.approx(start) == ds.utterances[utt_id].label_lists['phones'][index].start
        assert pytest.approx(end) == ds.utterances[utt_id].label_lists['phones'][index].end

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
