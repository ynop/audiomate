import os

from audiomate import corpus
from audiomate import issuers
from audiomate.corpus import io

from tests import resources
from . import reader_test as rt


class TestTudaReader(rt.CorpusReaderTest):

    # SHORT aliases for speakers
    SPK_cf = 'cf372280-5606-4b05-9d24-3ab7805d8462'
    SPK_55 = '55065c47-1290-4974-997e-e77f24e7c72d'
    SPK_75 = '755d9b71-f36e-45a6-a437-edebcfaee08d'
    SPK_9e = '9e6a00c9-80f0-479d-8b36-4139a9571217'
    SPK_58 = '58b8b441-684f-4753-aa16-589f1e149fa0'
    SPK_2a = '2a0995a7-47d8-453f-9864-5940efd3c71a'
    SPK_40 = '40a95aaf-2d87-43dc-b00e-9a4ceb77c6db'

    SAMPLE_PATH = resources.sample_corpus_path('tuda')
    FILE_TRACK_BASE_PATH = SAMPLE_PATH

    EXPECTED_NUMBER_OF_TRACKS = 48
    EXPECTED_TRACKS = [
        # DEV
        rt.ExpFileTrack('2015-01-27-11-31-32_Kinect-Beam', 'dev/2015-01-27-11-31-32_Kinect-Beam.wav'),
        rt.ExpFileTrack('2015-01-27-11-31-32_Kinect-RAW', 'dev/2015-01-27-11-31-32_Kinect-RAW.wav'),
        rt.ExpFileTrack('2015-01-27-11-31-32_Realtek', 'dev/2015-01-27-11-31-32_Realtek.wav'),
        rt.ExpFileTrack('2015-01-27-11-31-32_Samson', 'dev/2015-01-27-11-31-32_Samson.wav'),
        rt.ExpFileTrack('2015-01-28-12-36-24_Yamaha', 'dev/2015-01-28-12-36-24_Yamaha.wav'),
        # TEST
        rt.ExpFileTrack('2015-01-27-12-34-36_Kinect-Beam', 'test/2015-01-27-12-34-36_Kinect-Beam.wav'),
        rt.ExpFileTrack('2015-01-27-12-34-36_Kinect-RAW', 'test/2015-01-27-12-34-36_Kinect-RAW.wav'),
        rt.ExpFileTrack('2015-01-27-12-34-36_Realtek', 'test/2015-01-27-12-34-36_Realtek.wav'),
        rt.ExpFileTrack('2015-01-27-12-34-36_Samson', 'test/2015-01-27-12-34-36_Samson.wav'),
        rt.ExpFileTrack('2015-01-27-12-34-36_Yamaha', 'test/2015-01-27-12-34-36_Yamaha.wav'),
        # TRAIN
        rt.ExpFileTrack('2014-03-17-10-26-07_Microsoft-Kinect-Raw',
                        'train/2014-03-17-10-26-07_Microsoft-Kinect-Raw.wav'),
        rt.ExpFileTrack('2014-03-17-10-26-07_Realtek', 'train/2014-03-17-10-26-07_Realtek.wav'),
        rt.ExpFileTrack('2014-03-17-10-26-07_Yamaha', 'train/2014-03-17-10-26-07_Yamaha.wav'),
        rt.ExpFileTrack('2014-03-17-13-03-33_Kinect-Beam', 'train/2014-03-17-13-03-33_Kinect-Beam.wav'),
        rt.ExpFileTrack('2014-03-17-13-03-33_Realtek', 'train/2014-03-17-13-03-33_Realtek.wav'),
        rt.ExpFileTrack('2014-03-17-13-03-33_Yamaha', 'train/2014-03-17-13-03-33_Yamaha.wav'),
        rt.ExpFileTrack('2014-03-19-15-01-56_Kinect-Beam', 'train/2014-03-19-15-01-56_Kinect-Beam.wav'),
        # NOT ALL LISTED
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 7
    EXPECTED_ISSUERS = [
        rt.ExpSpeaker(SPK_cf, 8, issuers.Gender.MALE, issuers.AgeGroup.ADULT, 'deu'),
        rt.ExpSpeaker(SPK_55, 10, issuers.Gender.MALE, issuers.AgeGroup.ADULT, 'deu'),
        rt.ExpSpeaker(SPK_75, 7, issuers.Gender.MALE, issuers.AgeGroup.ADULT, 'deu'),
        rt.ExpSpeaker(SPK_9e, 5, issuers.Gender.FEMALE, issuers.AgeGroup.YOUTH, 'deu'),
        rt.ExpSpeaker(SPK_58, 10, issuers.Gender.FEMALE, issuers.AgeGroup.ADULT, 'deu'),
        rt.ExpSpeaker(SPK_2a, 5, issuers.Gender.MALE, issuers.AgeGroup.ADULT, 'deu'),
        rt.ExpSpeaker(SPK_40, 3, issuers.Gender.MALE, issuers.AgeGroup.ADULT, 'deu'),
    ]

    EXPECTED_NUMBER_OF_UTTERANCES = 48
    EXPECTED_UTTERANCES = [
        # DEV
        rt.ExpUtterance('2015-01-27-11-31-32_Kinect-Beam', '2015-01-27-11-31-32_Kinect-Beam', SPK_75, 0, float('inf')),
        rt.ExpUtterance('2015-01-27-11-31-32_Kinect-RAW', '2015-01-27-11-31-32_Kinect-RAW', SPK_75, 0, float('inf')),
        rt.ExpUtterance('2015-01-27-11-31-32_Realtek', '2015-01-27-11-31-32_Realtek', SPK_75, 0, float('inf')),
        rt.ExpUtterance('2015-01-27-11-31-32_Samson', '2015-01-27-11-31-32_Samson', SPK_75, 0, float('inf')),
        rt.ExpUtterance('2015-01-28-12-36-24_Yamaha', '2015-01-28-12-36-24_Yamaha', SPK_9e, 0, float('inf')),
        # TEST
        rt.ExpUtterance('2015-01-27-12-34-36_Kinect-Beam', '2015-01-27-12-34-36_Kinect-Beam', SPK_58, 0, float('inf')),
        rt.ExpUtterance('2015-01-27-12-34-36_Kinect-RAW', '2015-01-27-12-34-36_Kinect-RAW', SPK_58, 0, float('inf')),
        rt.ExpUtterance('2015-01-27-12-34-36_Realtek', '2015-01-27-12-34-36_Realtek', SPK_58, 0, float('inf')),
        rt.ExpUtterance('2015-01-27-12-34-36_Samson', '2015-01-27-12-34-36_Samson', SPK_58, 0, float('inf')),
        rt.ExpUtterance('2015-01-27-12-34-36_Yamaha', '2015-01-27-12-34-36_Yamaha', SPK_58, 0, float('inf')),
        # TRAIN
        rt.ExpUtterance('2014-03-17-10-26-07_Microsoft-Kinect-Raw',
                        '2014-03-17-10-26-07_Microsoft-Kinect-Raw', SPK_40, 0, float('inf')),
        rt.ExpUtterance('2014-03-17-10-26-07_Realtek', '2014-03-17-10-26-07_Realtek', SPK_40, 0, float('inf')),
        rt.ExpUtterance('2014-03-17-10-26-07_Yamaha', '2014-03-17-10-26-07_Yamaha', SPK_40, 0, float('inf')),
        rt.ExpUtterance('2014-03-17-13-03-33_Kinect-Beam', '2014-03-17-13-03-33_Kinect-Beam', SPK_cf, 0, float('inf')),
        rt.ExpUtterance('2014-03-17-13-03-33_Realtek', '2014-03-17-13-03-33_Realtek', SPK_cf, 0, float('inf')),
        rt.ExpUtterance('2014-03-17-13-03-33_Yamaha', '2014-03-17-13-03-33_Yamaha', SPK_cf, 0, float('inf')),
        rt.ExpUtterance('2014-03-19-15-01-56_Kinect-Beam', '2014-03-19-15-01-56_Kinect-Beam', SPK_cf, 0, float('inf')),
        # NOT ALL LISTED
    ]

    EXPECTED_LABEL_LISTS = {
        '2015-01-27-11-31-32_Samson': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT_RAW, 1),
        ],
        '2015-01-27-11-31-32_Realtek': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT_RAW, 1),
        ],
        '2014-03-17-10-26-07_Realtek': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT_RAW, 1),
        ],
        # NOT ALL LISTED
    }

    EXPECTED_LABELS = {
        '2015-01-27-11-31-32_Samson': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'Manche haben dass', 0, float('inf')),
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT_RAW, 'Manche haben , dass.', 0, float('inf')),
        ],
        '2015-01-27-11-31-32_Realtek': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'Manche haben dass', 0, float('inf')),
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT_RAW, 'Manche haben , dass.', 0, float('inf')),
        ],
        '2014-03-17-13-03-33_Realtek': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'Ich habe mich', 0, float('inf')),
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT_RAW, 'Ich habe mich.', 0, float('inf')),
        ],
        # NOT ALL LISTED
    }

    EXPECTED_NUMBER_OF_SUBVIEWS = 3 + 6 + 5 + 5 + 6
    EXPECTED_SUBVIEWS = [
        rt.ExpSubview('test', [
            '2015-01-27-12-34-36_Kinect-Beam',
            '2015-01-27-12-34-36_Kinect-RAW',
            '2015-01-27-12-34-36_Realtek',
            '2015-01-27-12-34-36_Samson',
            '2015-01-27-12-34-36_Yamaha',
            '2015-02-03-12-08-13_Kinect-Beam',
            '2015-02-03-12-08-13_Kinect-RAW',
            '2015-02-03-12-08-13_Realtek',
            '2015-02-03-12-08-13_Samson',
            '2015-02-03-12-08-13_Yamaha',
            '2015-02-10-14-31-52_Kinect-Beam',
            '2015-02-10-14-31-52_Kinect-RAW',
            '2015-02-10-14-31-52_Realtek',
            '2015-02-10-14-31-52_Samson',
            '2015-02-10-14-31-52_Yamaha',
        ]),
        rt.ExpSubview('Microsoft-Kinect-Raw', [
            '2014-03-17-10-26-07_Microsoft-Kinect-Raw',
            '2014-03-17-13-03-33_Microsoft-Kinect-Raw',
        ]),
        rt.ExpSubview('train_Samson', [
            '2014-08-07-13-22-38_Samson',
            '2014-08-14-14-52-00_Samson',
        ]),
        # NOT ALL LISTED
    ]

    def load(self):
        reader = io.TudaReader()
        return reader.load(self.SAMPLE_PATH)

    def test_get_ids_from_folder(self):
        assert io.TudaReader.get_ids_from_folder(os.path.join(self.SAMPLE_PATH, 'train'), 'train') == {
            '2014-03-17-10-26-07',
            '2014-03-17-13-03-33',
            '2014-03-19-15-01-56',
            '2014-08-07-13-22-38',
            '2014-08-14-14-52-00',
            '2015-04-17-11-26-07',
        }
        assert io.TudaReader.get_ids_from_folder(os.path.join(self.SAMPLE_PATH, 'dev'), 'dev') == {
            '2015-01-27-11-31-32',
            '2015-01-28-11-35-47',
            '2015-01-28-12-36-24',
        }
        assert io.TudaReader.get_ids_from_folder(os.path.join(self.SAMPLE_PATH, 'test'), 'test') == {
            '2015-01-27-12-34-36',
            '2015-02-03-12-08-13',
            '2015-02-10-14-31-52',
        }

    def test_get_ids_from_folder_ignore_bad_files(self):
        ids = io.TudaReader.get_ids_from_folder(os.path.join(self.SAMPLE_PATH, 'train'), 'train')
        assert '2014-08-05-11-08-34' not in ids

        ids = io.TudaReader.get_ids_from_folder(os.path.join(self.SAMPLE_PATH, 'dev'), 'dev')
        assert '2015-01-28-11-49-53' not in ids
