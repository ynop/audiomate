import os

import pytest
import requests_mock

from audiomate import corpus
from audiomate import issuers
from audiomate.corpus import io
from audiomate.corpus.io import mailabs

from tests import resources
from . import reader_test as rt


@pytest.fixture()
def tar_data():
    with open(resources.get_resource_path(['sample_files', 'cv_corpus_v1.tar.gz']), 'rb') as f:
        return f.read()


class TestMailabsDownloader:

    def test_download(self, tar_data, tmpdir):
        target_folder = tmpdir.strpath
        downloader = mailabs.MailabsDownloader(tags='de_DE')

        with requests_mock.Mocker() as mock:
            mock.get(mailabs.DOWNLOAD_URLS['de_DE'], content=tar_data)
            downloader.download(target_folder)

        base_path = os.path.join(target_folder, 'common_voice')

        assert os.path.isfile(os.path.join(base_path, 'cv-valid-dev.csv'))
        assert os.path.isdir(os.path.join(base_path, 'cv-valid-dev'))
        assert os.path.isfile(os.path.join(base_path, 'cv-valid-train.csv'))
        assert os.path.isdir(os.path.join(base_path, 'cv-valid-train'))


class TestMailabsReader(rt.CorpusReaderTest):

    SAMPLE_PATH = resources.sample_corpus_path('mailabs')
    FILE_TRACK_BASE_PATH = os.path.join(SAMPLE_PATH, 'files')

    EXPECTED_NUMBER_OF_TRACKS = 13
    EXPECTED_TRACKS = [
        rt.ExpFileTrack('elizabeth_klett-jane_eyre_01_f000001',
                        path=os.path.join(SAMPLE_PATH, 'en_US', 'by_book', 'female',
                                          'elizabeth_klett', 'jane_eyre',
                                          'wavs', 'jane_eyre_01_f000001.wav')),
        rt.ExpFileTrack('fred-azele_01_f000002',
                        path=os.path.join(SAMPLE_PATH, 'de_DE', 'by_book', 'male', 'fred',
                                          'azele', 'wavs', 'azele_01_f000002.wav')
                        ),
        rt.ExpFileTrack('abc_01_f000002',
                        path=os.path.join(SAMPLE_PATH, 'de_DE', 'by_book', 'mix',
                                          'abc', 'wavs', 'abc_01_f000002.wav')
                        ),
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 6
    EXPECTED_ISSUERS = [
        rt.ExpSpeaker('fred', 5, issuers.Gender.MALE, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('elizabeth_klett', 3, issuers.Gender.FEMALE, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('abc_01_f000002', 1, issuers.Gender.UNKNOWN, issuers.AgeGroup.UNKNOWN, None),
    ]

    EXPECTED_NUMBER_OF_UTTERANCES = 13
    EXPECTED_UTTERANCES = [
        rt.ExpUtterance('elizabeth_klett-jane_eyre_01_f000003', 'elizabeth_klett-jane_eyre_01_f000003',
                        'elizabeth_klett', 0, float('inf')),
        rt.ExpUtterance('sara-abc_01_f000001', 'sara-abc_01_f000001', 'sara', 0, float('inf')),
        rt.ExpUtterance('abc_01_f000001', 'abc_01_f000001', 'abc_01_f000001', 0, float('inf')),
    ]

    EXPECTED_LABEL_LISTS = {
        'elizabeth_klett-jane_eyre_01_f000003': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT_RAW, 1),
        ],
        'abc_01_f000002': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT_RAW, 1),
        ],
    }

    EXPECTED_LABELS = {
        'elizabeth_klett-jane_eyre_01_f000003': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT_RAW, 'Chapter 1.', 0, float('inf')),
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'Chapter one.', 0, float('inf')),
        ],
        'abc_01_f000002': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT_RAW, 'das 1.', 0, float('inf')),
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'das eins.', 0, float('inf')),
        ],
    }

    EXPECTED_NUMBER_OF_SUBVIEWS = 2
    EXPECTED_SUBVIEWS = [
        rt.ExpSubview('de_DE', [
            'fred-thego_01_f000001',
            'fred-thego_01_f000002',
            'fred-thego_01_f000003',
            'fred-azele_01_f000001',
            'fred-azele_01_f000002',
            'tim-azele_01_f000001',
            'sara-abc_01_f000001',
            'sara-abc_01_f000002',
            'abc_01_f000001',
            'abc_01_f000002',
        ]),
        rt.ExpSubview('en_US', [
            'elizabeth_klett-jane_eyre_01_f000001',
            'elizabeth_klett-jane_eyre_01_f000002',
            'elizabeth_klett-jane_eyre_01_f000003',
        ]),
    ]

    def load(self):
        return io.MailabsReader().load(self.SAMPLE_PATH)
