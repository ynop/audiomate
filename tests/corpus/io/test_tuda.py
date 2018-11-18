import os

import pytest

from audiomate import corpus
from audiomate import issuers
from audiomate.corpus import io
from tests import resources


@pytest.fixture
def reader():
    return io.TudaReader()


@pytest.fixture
def data_path():
    return resources.sample_corpus_path('tuda')


class TestTudaReader:

    def test_get_ids_from_folder(self, data_path):
        assert io.TudaReader.get_ids_from_folder(os.path.join(data_path, 'train'), 'train') == {
            '2014-03-17-13-03-33',
            '2014-03-19-15-01-56',
            '2014-08-07-13-22-38',
            '2014-08-14-14-52-00'
        }
        assert io.TudaReader.get_ids_from_folder(os.path.join(data_path, 'dev'), 'dev') == {
            '2015-01-27-11-31-32',
            '2015-01-28-11-35-47',
            '2015-01-28-12-36-24'
        }
        assert io.TudaReader.get_ids_from_folder(os.path.join(data_path, 'test'), 'test') == {
            '2015-01-27-12-34-36',
            '2015-02-03-12-08-13',
            '2015-02-10-14-31-52'
        }

    def test_get_ids_from_folder_ignore_bad_files(self, data_path):
        ids = io.TudaReader.get_ids_from_folder(os.path.join(data_path, 'train'), 'train')
        assert '2014-08-05-11-08-34' not in ids

        ids = io.TudaReader.get_ids_from_folder(os.path.join(data_path, 'dev'), 'dev')
        assert '2015-01-28-11-49-53' not in ids

    def test_load_tracks(self, reader, data_path):
        ds = reader.load(data_path)

        dev = os.path.join(data_path, 'dev')
        test = os.path.join(data_path, 'test')
        train = os.path.join(data_path, 'train')

        assert ds.num_tracks == 10

        assert ds.tracks['2014-03-17-13-03-33'].idx == '2014-03-17-13-03-33'
        assert ds.tracks['2014-03-17-13-03-33'].path == os.path.join(train, '2014-03-17-13-03-33_Kinect-Beam.wav')

        assert ds.tracks['2014-03-19-15-01-56'].idx == '2014-03-19-15-01-56'
        assert ds.tracks['2014-03-19-15-01-56'].path == os.path.join(train, '2014-03-19-15-01-56_Kinect-Beam.wav')

        assert ds.tracks['2014-08-07-13-22-38'].idx == '2014-08-07-13-22-38'
        assert ds.tracks['2014-08-07-13-22-38'].path == os.path.join(train, '2014-08-07-13-22-38_Kinect-Beam.wav')

        assert ds.tracks['2014-08-14-14-52-00'].idx == '2014-08-14-14-52-00'
        assert ds.tracks['2014-08-14-14-52-00'].path == os.path.join(train, '2014-08-14-14-52-00_Kinect-Beam.wav')

        assert ds.tracks['2015-01-27-11-31-32'].idx == '2015-01-27-11-31-32'
        assert ds.tracks['2015-01-27-11-31-32'].path == os.path.join(dev, '2015-01-27-11-31-32_Kinect-Beam.wav')

        assert ds.tracks['2015-01-28-11-35-47'].idx == '2015-01-28-11-35-47'
        assert ds.tracks['2015-01-28-11-35-47'].path == os.path.join(dev, '2015-01-28-11-35-47_Kinect-Beam.wav')

        assert ds.tracks['2015-01-28-12-36-24'].idx == '2015-01-28-12-36-24'
        assert ds.tracks['2015-01-28-12-36-24'].path == os.path.join(dev, '2015-01-28-12-36-24_Kinect-Beam.wav')

        assert ds.tracks['2015-01-27-12-34-36'].idx == '2015-01-27-12-34-36'
        assert ds.tracks['2015-01-27-12-34-36'].path == os.path.join(test, '2015-01-27-12-34-36_Kinect-Beam.wav')

        assert ds.tracks['2015-02-03-12-08-13'].idx == '2015-02-03-12-08-13'
        assert ds.tracks['2015-02-03-12-08-13'].path == os.path.join(test, '2015-02-03-12-08-13_Kinect-Beam.wav')

        assert ds.tracks['2015-02-10-14-31-52'].idx == '2015-02-10-14-31-52'
        assert ds.tracks['2015-02-10-14-31-52'].path == os.path.join(test, '2015-02-10-14-31-52_Kinect-Beam.wav')

    def test_load_speakers(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_issuers == 6

        assert ds.issuers['cf372280-5606-4b05-9d24-3ab7805d8462'].idx == 'cf372280-5606-4b05-9d24-3ab7805d8462'
        assert type(ds.issuers['cf372280-5606-4b05-9d24-3ab7805d8462']) == issuers.Speaker
        assert ds.issuers['cf372280-5606-4b05-9d24-3ab7805d8462'].gender == issuers.Gender.MALE
        assert ds.issuers['cf372280-5606-4b05-9d24-3ab7805d8462'].age_group == issuers.AgeGroup.ADULT
        assert ds.issuers['cf372280-5606-4b05-9d24-3ab7805d8462'].native_language == 'deu'

        assert ds.issuers['55065c47-1290-4974-997e-e77f24e7c72d'].idx == '55065c47-1290-4974-997e-e77f24e7c72d'
        assert type(ds.issuers['55065c47-1290-4974-997e-e77f24e7c72d']) == issuers.Speaker
        assert ds.issuers['55065c47-1290-4974-997e-e77f24e7c72d'].gender == issuers.Gender.MALE
        assert ds.issuers['55065c47-1290-4974-997e-e77f24e7c72d'].age_group == issuers.AgeGroup.ADULT
        assert ds.issuers['55065c47-1290-4974-997e-e77f24e7c72d'].native_language == 'deu'

        assert ds.issuers['755d9b71-f36e-45a6-a437-edebcfaee08d'].idx == '755d9b71-f36e-45a6-a437-edebcfaee08d'
        assert type(ds.issuers['755d9b71-f36e-45a6-a437-edebcfaee08d']) == issuers.Speaker
        assert ds.issuers['755d9b71-f36e-45a6-a437-edebcfaee08d'].gender == issuers.Gender.MALE
        assert ds.issuers['755d9b71-f36e-45a6-a437-edebcfaee08d'].age_group == issuers.AgeGroup.ADULT
        assert ds.issuers['755d9b71-f36e-45a6-a437-edebcfaee08d'].native_language == 'deu'

        assert ds.issuers['9e6a00c9-80f0-479d-8b36-4139a9571217'].idx == '9e6a00c9-80f0-479d-8b36-4139a9571217'
        assert type(ds.issuers['9e6a00c9-80f0-479d-8b36-4139a9571217']) == issuers.Speaker
        assert ds.issuers['9e6a00c9-80f0-479d-8b36-4139a9571217'].gender == issuers.Gender.FEMALE
        assert ds.issuers['9e6a00c9-80f0-479d-8b36-4139a9571217'].age_group == issuers.AgeGroup.YOUTH
        assert ds.issuers['9e6a00c9-80f0-479d-8b36-4139a9571217'].native_language == 'deu'

        assert ds.issuers['58b8b441-684f-4753-aa16-589f1e149fa0'].idx == '58b8b441-684f-4753-aa16-589f1e149fa0'
        assert type(ds.issuers['58b8b441-684f-4753-aa16-589f1e149fa0']) == issuers.Speaker
        assert ds.issuers['58b8b441-684f-4753-aa16-589f1e149fa0'].gender == issuers.Gender.FEMALE
        assert ds.issuers['58b8b441-684f-4753-aa16-589f1e149fa0'].age_group == issuers.AgeGroup.ADULT
        assert ds.issuers['58b8b441-684f-4753-aa16-589f1e149fa0'].native_language == 'deu'

        assert ds.issuers['2a0995a7-47d8-453f-9864-5940efd3c71a'].idx == '2a0995a7-47d8-453f-9864-5940efd3c71a'
        assert type(ds.issuers['2a0995a7-47d8-453f-9864-5940efd3c71a']) == issuers.Speaker
        assert ds.issuers['2a0995a7-47d8-453f-9864-5940efd3c71a'].gender == issuers.Gender.MALE
        assert ds.issuers['2a0995a7-47d8-453f-9864-5940efd3c71a'].age_group == issuers.AgeGroup.ADULT
        assert ds.issuers['2a0995a7-47d8-453f-9864-5940efd3c71a'].native_language == 'deu'

    def test_load_utterances(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_utterances == 10

        assert ds.utterances['2014-03-17-13-03-33'].idx == '2014-03-17-13-03-33'
        assert ds.utterances['2014-03-17-13-03-33'].track.idx == '2014-03-17-13-03-33'

        assert ds.utterances['2014-03-19-15-01-56'].idx == '2014-03-19-15-01-56'
        assert ds.utterances['2014-03-19-15-01-56'].track.idx == '2014-03-19-15-01-56'

        assert ds.utterances['2014-08-07-13-22-38'].idx == '2014-08-07-13-22-38'
        assert ds.utterances['2014-08-07-13-22-38'].track.idx == '2014-08-07-13-22-38'

        assert ds.utterances['2014-08-14-14-52-00'].idx == '2014-08-14-14-52-00'
        assert ds.utterances['2014-08-14-14-52-00'].track.idx == '2014-08-14-14-52-00'

        assert ds.utterances['2015-01-27-11-31-32'].idx == '2015-01-27-11-31-32'
        assert ds.utterances['2015-01-27-11-31-32'].track.idx == '2015-01-27-11-31-32'

        assert ds.utterances['2015-01-28-11-35-47'].idx == '2015-01-28-11-35-47'
        assert ds.utterances['2015-01-28-11-35-47'].track.idx == '2015-01-28-11-35-47'

        assert ds.utterances['2015-01-28-12-36-24'].idx == '2015-01-28-12-36-24'
        assert ds.utterances['2015-01-28-12-36-24'].track.idx == '2015-01-28-12-36-24'

        assert ds.utterances['2015-01-27-12-34-36'].idx == '2015-01-27-12-34-36'
        assert ds.utterances['2015-01-27-12-34-36'].track.idx == '2015-01-27-12-34-36'

        assert ds.utterances['2015-02-03-12-08-13'].idx == '2015-02-03-12-08-13'
        assert ds.utterances['2015-02-03-12-08-13'].track.idx == '2015-02-03-12-08-13'

        assert ds.utterances['2015-02-10-14-31-52'].idx == '2015-02-10-14-31-52'
        assert ds.utterances['2015-02-10-14-31-52'].track.idx == '2015-02-10-14-31-52'

    def test_load_transcriptions(self, reader, data_path):
        ds = reader.load(data_path)

        utt = ds.utterances['2014-03-17-13-03-33']
        assert utt.label_lists[corpus.LL_WORD_TRANSCRIPT][0].value == 'Ich habe mich'
        assert utt.label_lists[corpus.LL_WORD_TRANSCRIPT_RAW][0].value == 'Ich habe mich.'

        utt = ds.utterances['2014-03-19-15-01-56']
        assert utt.label_lists[corpus.LL_WORD_TRANSCRIPT][0].value == 'Felsnester hoch'
        assert utt.label_lists[corpus.LL_WORD_TRANSCRIPT_RAW][0].value == 'Felsnester hoch.'

        utt = ds.utterances['2014-08-07-13-22-38']
        assert utt.label_lists[corpus.LL_WORD_TRANSCRIPT][0].value == 'Ja'
        assert utt.label_lists[corpus.LL_WORD_TRANSCRIPT_RAW][0].value == 'Ja.'

        utt = ds.utterances['2014-08-14-14-52-00']
        assert utt.label_lists[corpus.LL_WORD_TRANSCRIPT][0].value == 'Ihr graubraunes'
        assert utt.label_lists[corpus.LL_WORD_TRANSCRIPT_RAW][0].value == 'Ihr graubraunes.'

        utt = ds.utterances['2015-01-27-11-31-32']
        assert utt.label_lists[corpus.LL_WORD_TRANSCRIPT][0].value == 'Manche haben dass'
        assert utt.label_lists[corpus.LL_WORD_TRANSCRIPT_RAW][0].value == 'Manche haben , dass.'

        utt = ds.utterances['2015-01-28-11-35-47']
        assert utt.label_lists[corpus.LL_WORD_TRANSCRIPT][0].value == 'Juni neun'
        assert utt.label_lists[corpus.LL_WORD_TRANSCRIPT_RAW][0].value == 'Juni 1919.'

        utt = ds.utterances['2015-01-28-12-36-24']
        assert utt.label_lists[corpus.LL_WORD_TRANSCRIPT][0].value == 'enthalten'
        assert utt.label_lists[corpus.LL_WORD_TRANSCRIPT_RAW][0].value == 'enthalten.'

        utt = ds.utterances['2015-01-27-12-34-36']
        assert utt.label_lists[corpus.LL_WORD_TRANSCRIPT][0].value == 'Unabhängigkeit'
        assert utt.label_lists[corpus.LL_WORD_TRANSCRIPT_RAW][0].value == 'Unabhängigkeit .'

        utt = ds.utterances['2015-02-03-12-08-13']
        assert utt.label_lists[corpus.LL_WORD_TRANSCRIPT][0].value == 'Was los ist'
        assert utt.label_lists[corpus.LL_WORD_TRANSCRIPT_RAW][0].value == 'Was los ist?'

        utt = ds.utterances['2015-02-10-14-31-52']
        assert utt.label_lists[corpus.LL_WORD_TRANSCRIPT][0].value == 'Jetzt kennenlernen'
        assert utt.label_lists[corpus.LL_WORD_TRANSCRIPT_RAW][0].value == 'Jetzt kennenlernen.'

    def test_load_subviews(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_subviews == 3

        assert ds.subviews['train'].num_utterances == 4
        assert ds.subviews['dev'].num_utterances == 3
        assert ds.subviews['test'].num_utterances == 3
