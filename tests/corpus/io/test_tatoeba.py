import os

import pytest
import requests_mock

from audiomate import corpus
from audiomate import issuers
from audiomate.corpus import io
from audiomate.corpus.io import tatoeba

from tests import resources
from . import reader_test as rt


@pytest.fixture()
def sample_audio_list_path():
    return resources.get_resource_path(['sample_corpora', 'tatoeba_download', 'sentences_with_audio.csv'])


@pytest.fixture()
def sample_sentence_list_path():
    return resources.get_resource_path(['sample_corpora', 'tatoeba_download', 'sentences.csv'])


@pytest.fixture()
def sample_audio_list_tar_bz():
    with open(resources.get_resource_path(['sample_corpora', 'tatoeba_download', 'sentences_with_audio.tar.bz2']),
              'rb') as f:
        return f.read()


@pytest.fixture()
def sample_sentence_list_tar_bz():
    with open(resources.get_resource_path(['sample_corpora', 'tatoeba_download', 'sentences.tar.bz2']), 'rb') as f:
        return f.read()


@pytest.fixture()
def sample_audio_content():
    with open(resources.get_resource_path(['wav_files', 'wav_2.wav']), 'rb') as f:
        return f.read()


class TestTatoebaDownloader:

    def test_load_audio_list(self, sample_audio_list_path):
        downloader = io.TatoebaDownloader()
        entries = downloader._load_audio_list(sample_audio_list_path)

        assert len(entries) == 5

        assert entries['247'] == ['gretelen', 'CC BY-NC 4.0', None]
        assert entries['1881'] == ['CK', 'CC BY-NC-ND 3.0', 'http://www.manythings.org/tatoeba']
        assert entries['6286'] == ['Phoenix', 'CC BY-NC 4.0', None]
        assert entries['2952354'] == ['pencil', 'CC BY-NC 4.0', None]
        assert entries['6921520'] == ['CK', 'CC BY-NC-ND 3.0', 'http://www.manythings.org/tatoeba']

    def test_load_audio_list_all(self, sample_audio_list_path):
        downloader = io.TatoebaDownloader(include_empty_licence=True)
        entries = downloader._load_audio_list(sample_audio_list_path)

        assert len(entries) == 7

        assert entries['141'] == ['BraveSentry', None, None]
        assert entries['247'] == ['gretelen', 'CC BY-NC 4.0', None]
        assert entries['1355'] == ['Nero', None, None]
        assert entries['1881'] == ['CK', 'CC BY-NC-ND 3.0', 'http://www.manythings.org/tatoeba']
        assert entries['6286'] == ['Phoenix', 'CC BY-NC 4.0', None]
        assert entries['2952354'] == ['pencil', 'CC BY-NC 4.0', None]
        assert entries['6921520'] == ['CK', 'CC BY-NC-ND 3.0', 'http://www.manythings.org/tatoeba']

    def test_load_audio_list_filter_license(self, sample_audio_list_path):
        downloader = io.TatoebaDownloader(include_licenses=['CC BY-NC 4.0'])
        entries = downloader._load_audio_list(sample_audio_list_path)

        assert len(entries) == 3

        assert entries['247'] == ['gretelen', 'CC BY-NC 4.0', None]
        assert entries['6286'] == ['Phoenix', 'CC BY-NC 4.0', None]
        assert entries['2952354'] == ['pencil', 'CC BY-NC 4.0', None]

    def test_load_sentence_list(self, sample_sentence_list_path):
        downloader = io.TatoebaDownloader()
        entries = downloader._load_sentence_list(sample_sentence_list_path)

        assert len(entries) == 8

        assert entries['141'] == ['eng', 'I want you to tell me why you did that.']
        assert entries['247'] == ['fra', 'Comment ça, je suis trop vieille pour ce poste ?']
        assert entries['511'] == ['deu', 'Wer will heiße Schokolade?']
        assert entries['524'] == ['deu', 'Das ist zu teuer!']
        assert entries['1355'] == ['epo', 'Mi panikis la homojn.']
        assert entries['6286'] == ['deu', 'Ich denke, ich habe genug gehört.']
        assert entries['299609'] == ['eng', 'He washes his car at least once a week.']
        assert entries['6921520'] == ['ita', 'Ho una zia che abita a Osaka.']

    def test_load_sentence_list_filter_languages(self, sample_sentence_list_path):
        downloader = io.TatoebaDownloader(include_languages=['deu', 'eng'])
        entries = downloader._load_sentence_list(sample_sentence_list_path)

        assert len(entries) == 5

        assert entries['141'] == ['eng', 'I want you to tell me why you did that.']
        assert entries['511'] == ['deu', 'Wer will heiße Schokolade?']
        assert entries['524'] == ['deu', 'Das ist zu teuer!']
        assert entries['6286'] == ['deu', 'Ich denke, ich habe genug gehört.']
        assert entries['299609'] == ['eng', 'He washes his car at least once a week.']

    def test_download(self, sample_audio_list_tar_bz, sample_sentence_list_tar_bz, sample_audio_content, tmpdir):
        downloader = io.TatoebaDownloader()

        with requests_mock.Mocker() as mock:
            mock.get(tatoeba.AUDIO_LIST_URL, content=sample_audio_list_tar_bz)
            mock.get(tatoeba.SENTENCE_LIST_URL, content=sample_sentence_list_tar_bz)

            mock.get('https://audio.tatoeba.org/sentences/eng/141.mp3', content=sample_audio_content)
            mock.get('https://audio.tatoeba.org/sentences/fra/247.mp3', content=sample_audio_content)
            mock.get('https://audio.tatoeba.org/sentences/epo/1355.mp3', content=sample_audio_content)
            mock.get('https://audio.tatoeba.org/sentences/deu/6286.mp3', content=sample_audio_content)
            mock.get('https://audio.tatoeba.org/sentences/ita/6921520.mp3', content=sample_audio_content)

            downloader.download(tmpdir.strpath)

            assert os.path.isfile(os.path.join(tmpdir.strpath, 'meta.txt'))

            assert not os.path.isfile(os.path.join(tmpdir.strpath, 'audio', 'eng', '141.mp3'))
            assert os.path.isfile(os.path.join(tmpdir.strpath, 'audio', 'fra', '247.mp3'))
            assert not os.path.isfile(os.path.join(tmpdir.strpath, 'audio', 'epo', '1355.mp3'))
            assert os.path.isfile(os.path.join(tmpdir.strpath, 'audio', 'deu', '6286.mp3'))
            assert os.path.isfile(os.path.join(tmpdir.strpath, 'audio', 'ita', '6921520.mp3'))

    def test_download_with_empty_licenses(self, sample_audio_list_tar_bz, sample_sentence_list_tar_bz,
                                          sample_audio_content, tmpdir):
        downloader = io.TatoebaDownloader(include_empty_licence=True)

        with requests_mock.Mocker() as mock:
            mock.get(tatoeba.AUDIO_LIST_URL, content=sample_audio_list_tar_bz)
            mock.get(tatoeba.SENTENCE_LIST_URL, content=sample_sentence_list_tar_bz)

            mock.get('https://audio.tatoeba.org/sentences/eng/141.mp3', content=sample_audio_content)
            mock.get('https://audio.tatoeba.org/sentences/fra/247.mp3', content=sample_audio_content)
            mock.get('https://audio.tatoeba.org/sentences/epo/1355.mp3', content=sample_audio_content)
            mock.get('https://audio.tatoeba.org/sentences/deu/6286.mp3', content=sample_audio_content)
            mock.get('https://audio.tatoeba.org/sentences/ita/6921520.mp3', content=sample_audio_content)

            downloader.download(tmpdir.strpath)

            assert os.path.isfile(os.path.join(tmpdir.strpath, 'meta.txt'))

            assert os.path.isfile(os.path.join(tmpdir.strpath, 'audio', 'eng', '141.mp3'))
            assert os.path.isfile(os.path.join(tmpdir.strpath, 'audio', 'fra', '247.mp3'))
            assert os.path.isfile(os.path.join(tmpdir.strpath, 'audio', 'epo', '1355.mp3'))
            assert os.path.isfile(os.path.join(tmpdir.strpath, 'audio', 'deu', '6286.mp3'))
            assert os.path.isfile(os.path.join(tmpdir.strpath, 'audio', 'ita', '6921520.mp3'))

    def test_download_with_filter_lang(self, sample_audio_list_tar_bz, sample_sentence_list_tar_bz,
                                       sample_audio_content, tmpdir):
        downloader = io.TatoebaDownloader(include_languages=['deu', 'eng'])

        with requests_mock.Mocker() as mock:
            mock.get(tatoeba.AUDIO_LIST_URL, content=sample_audio_list_tar_bz)
            mock.get(tatoeba.SENTENCE_LIST_URL, content=sample_sentence_list_tar_bz)

            mock.get('https://audio.tatoeba.org/sentences/eng/141.mp3', content=sample_audio_content)
            mock.get('https://audio.tatoeba.org/sentences/fra/247.mp3', content=sample_audio_content)
            mock.get('https://audio.tatoeba.org/sentences/epo/1355.mp3', content=sample_audio_content)
            mock.get('https://audio.tatoeba.org/sentences/deu/6286.mp3', content=sample_audio_content)
            mock.get('https://audio.tatoeba.org/sentences/ita/6921520.mp3', content=sample_audio_content)

            downloader.download(tmpdir.strpath)

            assert os.path.isfile(os.path.join(tmpdir.strpath, 'meta.txt'))

            assert not os.path.isfile(os.path.join(tmpdir.strpath, 'audio', 'eng', '141.mp3'))
            assert not os.path.isfile(os.path.join(tmpdir.strpath, 'audio', 'fra', '247.mp3'))
            assert not os.path.isfile(os.path.join(tmpdir.strpath, 'audio', 'epo', '1355.mp3'))
            assert os.path.isfile(os.path.join(tmpdir.strpath, 'audio', 'deu', '6286.mp3'))
            assert not os.path.isfile(os.path.join(tmpdir.strpath, 'audio', 'ita', '6921520.mp3'))

    def test_download_with_filter_license(self, sample_audio_list_tar_bz, sample_sentence_list_tar_bz,
                                          sample_audio_content, tmpdir):
        downloader = io.TatoebaDownloader(include_licenses=['CC BY-NC-ND 3.0'])

        with requests_mock.Mocker() as mock:
            mock.get(tatoeba.AUDIO_LIST_URL, content=sample_audio_list_tar_bz)
            mock.get(tatoeba.SENTENCE_LIST_URL, content=sample_sentence_list_tar_bz)

            mock.get('https://audio.tatoeba.org/sentences/eng/141.mp3', content=sample_audio_content)
            mock.get('https://audio.tatoeba.org/sentences/fra/247.mp3', content=sample_audio_content)
            mock.get('https://audio.tatoeba.org/sentences/epo/1355.mp3', content=sample_audio_content)
            mock.get('https://audio.tatoeba.org/sentences/deu/6286.mp3', content=sample_audio_content)
            mock.get('https://audio.tatoeba.org/sentences/ita/6921520.mp3', content=sample_audio_content)

            downloader.download(tmpdir.strpath)

            assert os.path.isfile(os.path.join(tmpdir.strpath, 'meta.txt'))

            assert not os.path.isfile(os.path.join(tmpdir.strpath, 'audio', 'eng', '141.mp3'))
            assert not os.path.isfile(os.path.join(tmpdir.strpath, 'audio', 'fra', '247.mp3'))
            assert not os.path.isfile(os.path.join(tmpdir.strpath, 'audio', 'epo', '1355.mp3'))
            assert not os.path.isfile(os.path.join(tmpdir.strpath, 'audio', 'deu', '6286.mp3'))
            assert os.path.isfile(os.path.join(tmpdir.strpath, 'audio', 'ita', '6921520.mp3'))


class TestTatoebaReader:

    SAMPLE_PATH = resources.sample_corpus_path('tatoeba')
    FILE_TRACK_BASE_PATH = os.path.join(SAMPLE_PATH, 'audio')

    EXPECTED_NUMBER_OF_TRACKS = 5
    EXPECTED_TRACKS = [
        rt.ExpFileTrack('141', os.path.join('eng', '141.mp3')),
        rt.ExpFileTrack('247', os.path.join('fra', '247.mp3')),
        rt.ExpFileTrack('1355', os.path.join('deu', '1355.mp3')),
        rt.ExpFileTrack('1881', os.path.join('deu', '1881.mp3')),
        rt.ExpFileTrack('6921520', os.path.join('eng', '6921520.mp3')),
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 4
    EXPECTED_ISSUERS = [
        rt.ExpSpeaker('BraveSentry', 1, issuers.Gender.UNKNOWN, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('gretelen', 1, issuers.Gender.UNKNOWN, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('Nero', 1, issuers.Gender.UNKNOWN, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('CK', 2, issuers.Gender.UNKNOWN, issuers.AgeGroup.UNKNOWN, None),
    ]

    EXPECTED_NUMBER_OF_UTTERANCES = 5
    EXPECTED_UTTERANCES = [
        rt.ExpUtterance('141', '141', 'BraveSentry', 0, float('inf')),
        rt.ExpUtterance('247', '247', 'gretelen', 0, float('inf')),
        rt.ExpUtterance('1355', '1355', 'Nero', 0, float('inf')),
        rt.ExpUtterance('1881', '1881', 'CK', 0, float('inf')),
        rt.ExpUtterance('6921520', '6921520', 'CK', 0, float('inf')),
    ]

    EXPECTED_LABEL_LISTS = {
        '141': [rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT_RAW, 1)],
        '247': [rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT_RAW, 1)],
        '1355': [rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT_RAW, 1)],
        '1881': [rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT_RAW, 1)],
        '6921520': [rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT_RAW, 1)],
    }

    EXPECTED_LABELS = {
        '141': [
            rt.ExpLabel(
                corpus.LL_WORD_TRANSCRIPT, 'I want you to tell me why you did that.',
                0, float('inf')
            )
        ],
        '247': [
            rt.ExpLabel(
                corpus.LL_WORD_TRANSCRIPT, 'Comment ça, je suis trop vieille pour ce poste ?',
                0, float('inf')
            )
        ],
        '1355': [
            rt.ExpLabel(
                corpus.LL_WORD_TRANSCRIPT, 'Wer will heiße Schokolade?',
                0, float('inf')
            )
        ],
        '1881': [
            rt.ExpLabel(
                corpus.LL_WORD_TRANSCRIPT, 'Das ist zu teuer!',
                0, float('inf')
            )
        ],
        '6921520': [
            rt.ExpLabel(
                corpus.LL_WORD_TRANSCRIPT, 'He washes his car at least once a week.',
                0, float('inf')
            )
        ],

    }

    EXPECTED_NUMBER_OF_SUBVIEWS = 0

    def load(self):
        return io.TatoebaReader().load(self.SAMPLE_PATH)
