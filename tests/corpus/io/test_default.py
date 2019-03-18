import os

import pytest

from audiomate import issuers
from audiomate.corpus import io
from audiomate.utils import jsonfile

from tests import resources
from . import reader_test as rt


@pytest.fixture()
def writer():
    return io.DefaultWriter()


@pytest.fixture()
def sample_corpus():
    return resources.create_dataset()


class TestDefaultReader(rt.CorpusReaderTest):

    SAMPLE_PATH = resources.sample_corpus_path('default')
    FILE_TRACK_BASE_PATH = os.path.join(SAMPLE_PATH, 'files')

    EXPECTED_NUMBER_OF_TRACKS = 6
    EXPECTED_TRACKS = [
        rt.ExpFileTrack('file-1', 'wav_1.wav'),
        rt.ExpFileTrack('file-2', 'wav_2.wav'),
        rt.ExpFileTrack('file-3', 'wav_3.wav'),
        rt.ExpFileTrack('file-4', 'wav_4.wav'),
        rt.ExpContainerTrack('file-5', 'file-5', os.path.join(SAMPLE_PATH, 'audio')),
        rt.ExpContainerTrack('file-6', 'file-6', os.path.join(SAMPLE_PATH, 'audio')),
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 3
    EXPECTED_ISSUERS = [
        rt.ExpSpeaker('speaker-1', 2, issuers.Gender.MALE, issuers.AgeGroup.ADULT, 'deu'),
        rt.ExpArtist('speaker-2', 2, 'Ohooo'),
        rt.ExpIssuer('speaker-3', 1),
    ]

    EXPECTED_NUMBER_OF_UTTERANCES = 5
    EXPECTED_UTTERANCES = [
        rt.ExpUtterance('utt-1', 'file-1', 'speaker-1', 0, float('inf')),
        rt.ExpUtterance('utt-2', 'file-2', 'speaker-1', 0, float('inf')),
        rt.ExpUtterance('utt-3', 'file-3', 'speaker-2', 0, 1.5),
        rt.ExpUtterance('utt-4', 'file-3', 'speaker-2', 1.5, 2.5),
        rt.ExpUtterance('utt-5', 'file-4', 'speaker-3', 0, float('inf')),
    ]

    EXPECTED_LABEL_LISTS = {
        'utt-1': [
            rt.ExpLabelList('text', 3),
            rt.ExpLabelList('raw_text', 1),
        ],
        'utt-2': [
            rt.ExpLabelList('text', 3),
            rt.ExpLabelList('raw_text', 1),
        ],
        'utt-3': [
            rt.ExpLabelList('text', 3),
            rt.ExpLabelList('raw_text', 1),
        ],
        'utt-4': [
            rt.ExpLabelList('text', 3),
            rt.ExpLabelList('raw_text', 1),
        ],
        'utt-5': [
            rt.ExpLabelList('text', 3),
            rt.ExpLabelList('raw_text', 1),
        ],
    }

    EXPECTED_LABELS = {
        'utt-1': [
            rt.ExpLabel('raw_text', 'who am i?', 0, float('inf')),
        ],
        'utt-4': [
            rt.ExpLabel('text', 'who', 0, float('inf')),
            rt.ExpLabel('text', 'are', 0, float('inf')),
            rt.ExpLabel('text', 'they', 3.5, 4.2),
        ],
    }

    EXPECTED_NUMBER_OF_SUBVIEWS = 2
    EXPECTED_SUBVIEWS = [
        rt.ExpSubview('train', [
            'utt-1',
            'utt-2',
            'utt-3',
        ]),
        rt.ExpSubview('dev', [
            'utt-4',
            'utt-5',
        ]),
    ]

    def test_load_issuers(self):
        ds = self.load()

        assert len(ds.issuers['speaker-3'].info) == 1
        assert ds.issuers['speaker-3'].info['region'] == 'zh'

    def test_load_label_meta(self):
        ds = self.load()

        utt_2 = ds.utterances['utt-2']
        utt_3 = ds.utterances['utt-3']
        utt_4 = ds.utterances['utt-4']

        utt_2_labels = sorted(utt_2.label_lists['text'].labels)
        utt_3_labels = sorted(utt_3.label_lists['text'].labels)
        utt_4_labels = sorted(utt_4.label_lists['text'].labels)

        assert len(utt_2_labels[1].meta) == 3
        assert utt_2_labels[1].meta['pron'] == 'huu'
        assert utt_2_labels[1].meta['duration'] == 2.3
        assert utt_2_labels[1].meta['stressed']

        assert len(utt_3_labels[0].meta) == 0

        assert len(utt_4_labels[2].meta) == 1
        assert utt_4_labels[2].meta['ex.'] == 19

    def test_load_features(self):
        ds = self.load()

        assert ds.feature_containers['mfcc'].path == os.path.join(self.SAMPLE_PATH, 'features', 'mfcc')
        assert ds.feature_containers['fbank'].path == os.path.join(self.SAMPLE_PATH, 'features', 'fbank')

    def load(self):
        return io.DefaultReader().load(self.SAMPLE_PATH)


class TestDefaultWriter:

    def test_save_files_exist(self, writer, sample_corpus, tmpdir):
        writer.save(sample_corpus, tmpdir.strpath)
        files = os.listdir(tmpdir.strpath)

        assert len(files) == 9

        assert 'files.txt' in files
        assert 'audio.txt' in files
        assert 'issuers.json' in files
        assert 'utterances.txt' in files
        assert 'utt_issuers.txt' in files
        assert 'labels_word-transcript.txt' in files
        assert 'subview_train.txt' in files
        assert 'subview_dev.txt' in files

    def test_save_file_tracks(self, writer, sample_corpus, tmpdir):
        # make sure relative path changes in contrast to self.ds.path
        out_path = os.path.join(tmpdir.strpath, 'somesubdir')
        os.makedirs(out_path)

        writer.save(sample_corpus, out_path)

        file_1_path = os.path.relpath(resources.sample_wav_file('wav_1.wav'), out_path)
        file_2_path = os.path.relpath(resources.sample_wav_file('wav_2.wav'), out_path)
        file_3_path = os.path.relpath(resources.sample_wav_file('wav_3.wav'), out_path)
        file_4_path = os.path.relpath(resources.sample_wav_file('wav_4.wav'), out_path)

        with open(os.path.join(out_path, 'files.txt'), 'r') as f:
            file_content = f.read()

        assert file_content.strip() == 'wav-1 {}\nwav_2 {}\nwav_3 {}\nwav_4 {}'.format(file_1_path,
                                                                                       file_2_path,
                                                                                       file_3_path,
                                                                                       file_4_path)

    def test_save_container_tracks(self, writer, tmpdir):
        # make sure relative path changes in contrast to self.ds.path
        out_path = os.path.join(tmpdir.strpath, 'somesubdir')
        os.makedirs(out_path)

        sample_corpus = io.DefaultReader().load(TestDefaultReader.SAMPLE_PATH)

        writer.save(sample_corpus, out_path)

        with open(os.path.join(out_path, 'audio.txt'), 'r') as f:
            file_content = f.read()

        rel_path = os.path.relpath(os.path.join(TestDefaultReader.SAMPLE_PATH, 'audio'), out_path)
        assert file_content.strip() == '\n'.join((
            'file-5 {} file-5'.format(rel_path),
            'file-6 {} file-6'.format(rel_path)
        ))

    def test_save_issuers(self, writer, sample_corpus, tmpdir):
        writer.save(sample_corpus, tmpdir.strpath)
        data = jsonfile.read_json_file(os.path.join(tmpdir.strpath, 'issuers.json'))

        expected = {
            'spk-1': {
                'type': 'speaker',
                'gender': 'male'
            },
            'spk-2': {
                'type': 'speaker',
                'gender': 'female'
            },
            'spk-3': {}
        }

        assert data == expected

    def test_save_utterances(self, writer, sample_corpus, tmpdir):
        writer.save(sample_corpus, tmpdir.strpath)

        with open(os.path.join(tmpdir.strpath, 'utterances.txt'), 'r') as f:
            file_content = f.read()

        assert file_content.strip() == 'utt-1 wav-1 0 -1\n' \
                                       'utt-2 wav_2 0 -1\n' \
                                       'utt-3 wav_3 0 1.5\n' \
                                       'utt-4 wav_3 1.5 2.5\n' \
                                       'utt-5 wav_4 0 -1'

    def test_save_utt_to_issuer(self, writer, sample_corpus, tmpdir):
        writer.save(sample_corpus, tmpdir.strpath)

        with open(os.path.join(tmpdir.strpath, 'utt_issuers.txt'), 'r') as f:
            file_content = f.read()

        assert file_content.strip() == 'utt-1 spk-1\n' \
                                       'utt-2 spk-1\n' \
                                       'utt-3 spk-2\n' \
                                       'utt-4 spk-2\n' \
                                       'utt-5 spk-3'

    def test_save_labels(self, writer, sample_corpus, tmpdir):
        writer.save(sample_corpus, tmpdir.strpath)

        with open(os.path.join(tmpdir.strpath, 'labels_word-transcript.txt'), 'r') as f:
            file_content = f.read()

        assert file_content.strip() == 'utt-1 0 -1 who am i\n' \
                                       'utt-2 0 -1 who are you [{"a": "hey", "b": 2}]\n' \
                                       'utt-3 0 -1 who is he\n' \
                                       'utt-4 0 -1 who are they\n' \
                                       'utt-5 0 -1 who is she'

    def test_save_subviews(self, writer, sample_corpus, tmpdir):
        writer.save(sample_corpus, tmpdir.strpath)

        with open(os.path.join(tmpdir.strpath, 'subview_train.txt'), 'r') as f:
            sv_train_content = f.read()

        with open(os.path.join(tmpdir.strpath, 'subview_dev.txt'), 'r') as f:
            sv_dev_content = f.read()

        assert sv_train_content.strip() == 'matching_utterance_ids\ninclude,utt-1,utt-2,utt-3'
        assert sv_dev_content.strip() == 'matching_utterance_ids\ninclude,utt-4,utt-5'

    def test_save_utterances_with_no_issuer(self, writer, sample_corpus, tmpdir):
        sample_corpus.utterances['utt-3'].issuer = None
        sample_corpus.utterances['utt-4'].issuer = None
        sample_corpus.utterances['utt-5'].issuer = None

        writer.save(sample_corpus, tmpdir.strpath)

        with open(os.path.join(tmpdir.strpath, 'utterances.txt'), 'r') as f:
            file_content = f.read()

        assert file_content.strip() == 'utt-1 wav-1 0 -1\n' \
                                       'utt-2 wav_2 0 -1\n' \
                                       'utt-3 wav_3 0 1.5\n' \
                                       'utt-4 wav_3 1.5 2.5\n' \
                                       'utt-5 wav_4 0 -1'

        with open(os.path.join(tmpdir.strpath, 'utt_issuers.txt'), 'r') as f:
            file_content = f.read()

        assert file_content.strip() == 'utt-1 spk-1\n' \
                                       'utt-2 spk-1'
