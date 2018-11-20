import os

import pytest
import requests_mock

from audiomate import corpus
from audiomate import issuers
from audiomate.corpus.io import mailabs

from tests import resources


@pytest.fixture()
def reader():
    return mailabs.MailabsReader()


@pytest.fixture()
def tar_data():
    with open(resources.get_resource_path(['sample_files', 'cv_corpus_v1.tar.gz']), 'rb') as f:
        return f.read()


@pytest.fixture
def data_path():
    return resources.sample_corpus_path('mailabs')


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


class TestMailabsReader:

    def test_load_correct_number_of_tracks(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_tracks == 13

    def test_load_tracks(self, reader, data_path):
        ds = reader.load(data_path)

        idx = 'elizabeth_klett-jane_eyre_01_f000001'
        path = os.path.join(data_path, 'en_US', 'by_book', 'female',
                            'elizabeth_klett', 'jane_eyre',
                            'wavs', 'jane_eyre_01_f000001.wav')

        assert ds.tracks[idx].idx == idx
        assert ds.tracks[idx].path == path

        idx = 'fred-azele_01_f000002'
        path = os.path.join(data_path, 'de_DE', 'by_book', 'male', 'fred',
                            'azele', 'wavs', 'azele_01_f000002.wav')

        assert ds.tracks[idx].idx == idx
        assert ds.tracks[idx].path == path

        idx = 'abc_01_f000002'
        path = os.path.join(data_path, 'de_DE', 'by_book', 'mix',
                            'abc', 'wavs', 'abc_01_f000002.wav')

        assert ds.tracks[idx].idx == idx
        assert ds.tracks[idx].path == path

    def test_load_correct_number_of_issuers(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_issuers == 6

    def test_load_issuers(self, reader, data_path):
        ds = reader.load(data_path)

        idx = 'fred'

        assert ds.issuers[idx].idx == idx
        assert ds.issuers[idx].gender == issuers.Gender.MALE
        assert ds.issuers[idx].age_group == issuers.AgeGroup.UNKNOWN
        assert type(ds.issuers[idx]) == issuers.Speaker
        assert len(ds.issuers[idx].utterances) == 5

        idx = 'elizabeth_klett'

        assert ds.issuers[idx].idx == idx
        assert ds.issuers[idx].gender == issuers.Gender.FEMALE
        assert ds.issuers[idx].age_group == issuers.AgeGroup.UNKNOWN
        assert type(ds.issuers[idx]) == issuers.Speaker
        assert len(ds.issuers[idx].utterances) == 3

        idx = 'abc_01_f000002'

        assert ds.issuers[idx].idx == idx
        assert ds.issuers[idx].gender == issuers.Gender.UNKNOWN
        assert ds.issuers[idx].age_group == issuers.AgeGroup.UNKNOWN
        assert type(ds.issuers[idx]) == issuers.Speaker
        assert len(ds.issuers[idx].utterances) == 1

    def test_load_correct_number_of_utterances(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_utterances == 13

    def test_load_utterances(self, reader, data_path):
        ds = reader.load(data_path)

        idx = 'elizabeth_klett-jane_eyre_01_f000003'

        assert ds.utterances[idx].idx == idx
        assert ds.utterances[idx].track.idx == idx
        assert ds.utterances[idx].issuer.idx == 'elizabeth_klett'
        assert ds.utterances[idx].start == 0
        assert ds.utterances[idx].end == -1

        idx = 'sara-abc_01_f000001'

        assert ds.utterances[idx].idx == idx
        assert ds.utterances[idx].track.idx == idx
        assert ds.utterances[idx].issuer.idx == 'sara'
        assert ds.utterances[idx].start == 0
        assert ds.utterances[idx].end == -1

        idx = 'abc_01_f000001'

        assert ds.utterances[idx].idx == idx
        assert ds.utterances[idx].track.idx == idx
        assert ds.utterances[idx].issuer.idx == idx
        assert ds.utterances[idx].start == 0
        assert ds.utterances[idx].end == -1

    def test_load_transcription_raw(self, reader, data_path):
        ds = reader.load(data_path)
        ll_idx = corpus.LL_WORD_TRANSCRIPT_RAW

        utt = ds.utterances['elizabeth_klett-jane_eyre_01_f000003']
        ll = utt.label_lists[ll_idx]

        assert len(ll) == 1
        assert ll[0].value == 'Chapter 1.'
        assert ll[0].start == 0
        assert ll[0].end == -1

        utt = ds.utterances['abc_01_f000002']
        ll = utt.label_lists[ll_idx]

        assert len(ll) == 1
        assert ll[0].value == 'das 1.'
        assert ll[0].start == 0
        assert ll[0].end == -1

    def test_load_transcription_clean(self, reader, data_path):
        ds = reader.load(data_path)
        ll_idx = corpus.LL_WORD_TRANSCRIPT

        utt = ds.utterances['elizabeth_klett-jane_eyre_01_f000003']
        ll = utt.label_lists[ll_idx]

        assert len(ll) == 1
        assert ll[0].value == 'Chapter one.'
        assert ll[0].start == 0
        assert ll[0].end == -1

        utt = ds.utterances['abc_01_f000002']
        ll = utt.label_lists[ll_idx]

        assert len(ll) == 1
        assert ll[0].value == 'das eins.'
        assert ll[0].start == 0
        assert ll[0].end == -1

    def test_load_correct_number_of_subviews(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_subviews == 2

    def test_subviews(self, reader, data_path):
        ds = reader.load(data_path)

        assert 'de_DE' in ds.subviews.keys()
        assert ds.subviews['de_DE'].num_utterances == 10
        assert set(ds.subviews['de_DE'].utterances.keys()) == {
            'fred-thego_01_f000001',
            'fred-thego_01_f000002',
            'fred-thego_01_f000003',
            'fred-azele_01_f000001',
            'fred-azele_01_f000002',
            'tim-azele_01_f000001',
            'sara-abc_01_f000001',
            'sara-abc_01_f000002',
            'abc_01_f000001',
            'abc_01_f000002'
        }

        assert 'en_US' in ds.subviews.keys()
        assert ds.subviews['en_US'].num_utterances == 3
        assert set(ds.subviews['en_US'].utterances.keys()) == {
            'elizabeth_klett-jane_eyre_01_f000001',
            'elizabeth_klett-jane_eyre_01_f000002',
            'elizabeth_klett-jane_eyre_01_f000003'
        }

    def test_ignores_utterance_with_missing_wav(self, reader, data_path):
        ds = reader.load(data_path)
        assert 'tim-azele_01_f000002' not in ds.utterances.keys()
