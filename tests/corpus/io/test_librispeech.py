import os

from audiomate import corpus
from audiomate import issuers
from audiomate.corpus import io
from audiomate.corpus.io import librispeech

import requests_mock

from tests import resources
from . import reader_test as rt


class TestLibriSpeechDownloader:

    def test_download_all(self, tmpdir):
        target_folder = tmpdir.strpath
        downloader = io.LibriSpeechDownloader()

        with requests_mock.Mocker() as mock:
            # Return any size (doesn't matter, only for prints)
            for name, url in librispeech.SUBSETS.items():
                data_path = resources.get_resource_path([
                    'sample_archives',
                    'librispeech',
                    '{}.tar.gz'.format(name)
                ])
                with open(data_path, 'rb') as f:
                    data = f.read()
                mock.head(url, headers={'Content-Length': '100'})
                mock.get(url, content=data)

            downloader.download(target_folder)

        for name in librispeech.SUBSETS:
            assert os.path.isdir(os.path.join(target_folder, name))

        assert os.path.isfile(os.path.join(target_folder, 'BOOKS.TXT'))
        assert os.path.isfile(os.path.join(target_folder, 'CHAPTERS.TXT'))
        assert os.path.isfile(os.path.join(target_folder, 'LICENSE.TXT'))
        assert os.path.isfile(os.path.join(target_folder, 'README.TXT'))
        assert os.path.isfile(os.path.join(target_folder, 'SPEAKERS.TXT'))

    def test_download_two_subsets(self, tmpdir):
        target_folder = tmpdir.strpath
        downloader = io.LibriSpeechDownloader(subsets=['dev-clean', 'test-clean'])

        with requests_mock.Mocker() as mock:
            # Return any size (doesn't matter, only for prints)
            for name, url in librispeech.SUBSETS.items():
                data_path = resources.get_resource_path([
                    'sample_archives',
                    'librispeech',
                    '{}.tar.gz'.format(name)
                ])
                with open(data_path, 'rb') as f:
                    data = f.read()
                mock.head(url, headers={'Content-Length': '100'})
                mock.get(url, content=data)

            downloader.download(target_folder)

        for name in ['dev-clean', 'test-clean']:
            assert os.path.isdir(os.path.join(target_folder, name))

        assert os.path.isfile(os.path.join(target_folder, 'BOOKS.TXT'))
        assert os.path.isfile(os.path.join(target_folder, 'CHAPTERS.TXT'))
        assert os.path.isfile(os.path.join(target_folder, 'LICENSE.TXT'))
        assert os.path.isfile(os.path.join(target_folder, 'README.TXT'))
        assert os.path.isfile(os.path.join(target_folder, 'SPEAKERS.TXT'))


class TestLibriSpeechReader(rt.CorpusReaderTest):

    SAMPLE_PATH = resources.sample_corpus_path('librispeech')
    FILE_TRACK_BASE_PATH = os.path.join(SAMPLE_PATH)

    EXPECTED_NUMBER_OF_TRACKS = 12
    EXPECTED_TRACKS = [
        rt.ExpFileTrack('174-32-0001', 'dev-clean/174/32/174-32-0001.flac'),
        rt.ExpFileTrack('174-32-0002', 'dev-clean/174/32/174-32-0002.flac'),
        rt.ExpFileTrack('174-32-0003', 'dev-clean/174/32/174-32-0003.flac'),
        rt.ExpFileTrack('8842-12-0001', 'dev-clean/8842/12/8842-12-0001.flac'),
        rt.ExpFileTrack('8842-383-0001', 'dev-clean/8842/383/8842-383-0001.flac'),
        rt.ExpFileTrack('8842-383-0002', 'dev-clean/8842/383/8842-383-0002.flac'),
        rt.ExpFileTrack('14-32-0001', 'train-clean-360/14/32/14-32-0001.flac'),
        rt.ExpFileTrack('14-32-0002', 'train-clean-360/14/32/14-32-0002.flac'),
        rt.ExpFileTrack('14-384-0001', 'train-clean-360/14/384/14-384-0001.flac'),
        rt.ExpFileTrack('17-22-0001', 'train-clean-360/17/22/17-22-0001.flac'),
        rt.ExpFileTrack('17-22-0002', 'train-clean-360/17/22/17-22-0002.flac'),
        rt.ExpFileTrack('19-99-0001', 'train-clean-360/19/99/19-99-0001.flac'),
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 5
    EXPECTED_ISSUERS = [
        rt.ExpSpeaker('14', 3, issuers.Gender.FEMALE, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('17', 2, issuers.Gender.MALE, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('19', 1, issuers.Gender.FEMALE, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('8842', 3, issuers.Gender.FEMALE, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('174', 3, issuers.Gender.MALE, issuers.AgeGroup.UNKNOWN, None),
    ]

    EXPECTED_NUMBER_OF_UTTERANCES = 12
    EXPECTED_UTTERANCES = [
        rt.ExpUtterance('174-32-0001',   '174-32-0001',   '174',  0, float('inf')),
        rt.ExpUtterance('174-32-0002',   '174-32-0002',   '174',  0, float('inf')),
        rt.ExpUtterance('174-32-0003',   '174-32-0003',   '174',  0, float('inf')),
        rt.ExpUtterance('8842-12-0001',  '8842-12-0001',  '8842', 0, float('inf')),
        rt.ExpUtterance('8842-383-0001', '8842-383-0001', '8842', 0, float('inf')),
        rt.ExpUtterance('8842-383-0002', '8842-383-0002', '8842', 0, float('inf')),
        rt.ExpUtterance('14-32-0001',    '14-32-0001',    '14',   0, float('inf')),
        rt.ExpUtterance('14-32-0002',    '14-32-0002',    '14',   0, float('inf')),
        rt.ExpUtterance('14-384-0001',   '14-384-0001',   '14',   0, float('inf')),
        rt.ExpUtterance('17-22-0001',    '17-22-0001',    '17',   0, float('inf')),
        rt.ExpUtterance('17-22-0002',    '17-22-0002',    '17',   0, float('inf')),
        rt.ExpUtterance('19-99-0001',    '19-99-0001',    '19',   0, float('inf')),
    ]

    EXPECTED_LABEL_LISTS = {
        '174-32-0001': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
        ],
        '174-32-0002': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
        ],
        '174-32-0003': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
        ],
        '8842-12-0001': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1)
        ],
        '14-384-0001': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1)
        ],
    }

    EXPECTED_LABELS = {
        '174-32-0001': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'A', 0, float('inf')),
        ],
        '174-32-0002': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'B', 0, float('inf')),
        ],
        '174-32-0003': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'C', 0, float('inf')),
        ],
        '8842-12-0001': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'XY', 0, float('inf')),
        ],
        '14-384-0001': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'HOPE FOR', 0, float('inf')),
        ],
    }

    EXPECTED_NUMBER_OF_SUBVIEWS = 2

    EXPECTED_SUBVIEWS = [
        rt.ExpSubview('dev-clean', [
            '174-32-0001',
            '174-32-0002',
            '174-32-0003',
            '8842-12-0001',
            '8842-383-0001',
            '8842-383-0002',
        ]),
        rt.ExpSubview('train-clean-360', [
            '14-32-0001',
            '14-32-0002',
            '14-384-0001',
            '17-22-0001',
            '17-22-0002',
            '19-99-0001',
        ]),
    ]

    def load(self):
        reader = librispeech.LibriSpeechReader()
        return reader.load(self.SAMPLE_PATH)
