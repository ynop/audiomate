import os

import pytest

import audiomate
from audiomate.corpus import io
from audiomate import issuers
from tests import resources


@pytest.fixture
def reader():
    return io.SWCReader()


@pytest.fixture
def sample_path():
    return resources.sample_corpus_path('swc')


class TestSWCReader:

    def test_get_articles(self, reader, sample_path):
        articles = reader.get_articles(sample_path)

        article_names = {
            'Fonteius_Capito',
            'Kritischer_Rationalismus',
            'Philosophie',
            'Radio',
        }

        article_paths = {os.path.join(sample_path, a) for a in article_names}

        assert set(articles) == article_paths

    def test_get_audio_file_info(self, reader, sample_path):
        article_path = os.path.join(sample_path, 'Kritischer_Rationalismus')
        audio_files = reader.get_audio_file_info(article_path)

        assert audio_files == {
            os.path.join(article_path, 'audio1.ogg'): 0.0,
            os.path.join(article_path, 'audio2.ogg'): 1224.0,
            os.path.join(article_path, 'audio3.ogg'): 2713.114375,
        }

    def test_get_reader_info_male(self, reader, sample_path):
        article_path = os.path.join(sample_path, 'Kritischer_Rationalismus')
        name, gender = reader.get_reader_info(article_path)

        assert name == 'Jan Krüger'
        assert gender == issuers.Gender.MALE

    def test_get_reader_info_female(self, reader, sample_path):
        article_path = os.path.join(sample_path, 'Fonteius_Capito')
        name, gender = reader.get_reader_info(article_path)

        assert name == 'Souffleuse'
        assert gender == issuers.Gender.FEMALE

    def test_find_audio_file_for_segment_if_only_one_audio_file(self, reader):
        audio_files = {
            'a': 0
        }
        res = reader.find_audio_file_for_segment(291, 302, audio_files)

        assert res == 'a'

    def test_find_audio_file_for_segment_if_in_first_audio_file(self, reader):
        audio_files = {
            'a': 0,
            'b': 193,
            'c': 340,
        }
        res = reader.find_audio_file_for_segment(89, 193, audio_files)

        assert res == 'a'

    def test_find_audio_file_for_segment_if_in_intermediate_audio_file(self, reader):
        audio_files = {
            'a': 0,
            'b': 291,
            'c': 340,
        }
        res = reader.find_audio_file_for_segment(291, 302, audio_files)

        assert res == 'b'

    def test_find_audio_file_for_segment_if_in_last_audio_file(self, reader):
        audio_files = {
            'a': 0,
            'b': 221,
            'c': 291,
        }
        res = reader.find_audio_file_for_segment(291, 302, audio_files)

        assert res == 'c'

    def test_find_audio_file_for_segment_if_across_audio_files(self, reader):
        audio_files = {
            'a': 0,
            'b': 221,
            'c': 291,
        }
        res = reader.find_audio_file_for_segment(279, 302, audio_files)

        assert res is None

    def test_get_segments(self, reader, sample_path):
        article_path = os.path.join(sample_path, 'Fonteius_Capito')
        segments = reader.get_segments(article_path)

        exp = [
            (1.010, 3.620, 'Sie hören den Artikel Fonteius Capito'),
            (27.780, 30.680, 'Im Jahr siebenundsechzig nach Christus'),
            (73.530, 78.340, 'dem Tod Neros wurde Fonteius Capito wegen angeblicher Umsturzpläne'),
        ]

        assert sorted(segments, key=lambda x: x[0]) == exp

    def test_load(self, reader, sample_path):
        ds = reader.load(sample_path)

        assert ds.num_issuers == 4
        assert ds.num_utterances == 2055
        assert ds.num_tracks == 7

        utt = None

        for utterance in ds.utterances.values():
            if utterance.idx.endswith('4338630_4340650'):
                utt = utterance

        assert utt.start == 4338.63 - 2713.114375
        assert utt.end == 4340.650 - 2713.114375

        ll = utt.label_lists[audiomate.corpus.LL_WORD_TRANSCRIPT]

        assert len(ll) == 1
        assert ll.join() == 'Bartley vertrat die Auffassung'
