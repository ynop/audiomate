import os

import pytest

from audiomate import issuers
from audiomate.corpus import io
from audiomate.utils import jsonfile
from tests import resources


@pytest.fixture()
def reader():
    return io.DefaultReader()


@pytest.fixture()
def writer():
    return io.DefaultWriter()


@pytest.fixture()
def sample_corpus_path():
    return resources.sample_corpus_path('default')


@pytest.fixture()
def sample_corpus():
    return resources.create_dataset()


class TestDefaultReader:

    def test_load_tracks(self, reader, sample_corpus_path):
        ds = reader.load(sample_corpus_path)

        assert ds.num_tracks == 6
        assert ds.tracks['file-1'].idx == 'file-1'
        assert ds.tracks['file-1'].path == os.path.join(sample_corpus_path, 'files', 'wav_1.wav')
        assert ds.tracks['file-2'].idx == 'file-2'
        assert ds.tracks['file-2'].path == os.path.join(sample_corpus_path, 'files', 'wav_2.wav')
        assert ds.tracks['file-3'].idx == 'file-3'
        assert ds.tracks['file-3'].path == os.path.join(sample_corpus_path, 'files', 'wav_3.wav')
        assert ds.tracks['file-4'].idx == 'file-4'
        assert ds.tracks['file-4'].path == os.path.join(sample_corpus_path, 'files', 'wav_4.wav')
        assert ds.tracks['file-5'].idx == 'file-5'
        assert ds.tracks['file-5'].key == 'file-5'
        assert ds.tracks['file-5'].container.path == os.path.join(sample_corpus_path, 'audio')
        assert ds.tracks['file-6'].idx == 'file-6'
        assert ds.tracks['file-6'].key == 'file-6'
        assert ds.tracks['file-6'].container.path == os.path.join(sample_corpus_path, 'audio')

    def test_load_utterances(self, reader, sample_corpus_path):
        ds = reader.load(sample_corpus_path)

        assert ds.num_utterances == 5

        assert ds.utterances['utt-1'].idx == 'utt-1'
        assert ds.utterances['utt-1'].track.idx == 'file-1'
        assert ds.utterances['utt-1'].issuer.idx == 'speaker-1'
        assert ds.utterances['utt-1'].start == 0
        assert ds.utterances['utt-1'].end == -1

        assert ds.utterances['utt-2'].idx == 'utt-2'
        assert ds.utterances['utt-2'].track.idx == 'file-2'
        assert ds.utterances['utt-2'].issuer.idx == 'speaker-1'
        assert ds.utterances['utt-2'].start == 0
        assert ds.utterances['utt-2'].end == -1

        assert ds.utterances['utt-3'].idx == 'utt-3'
        assert ds.utterances['utt-3'].track.idx == 'file-3'
        assert ds.utterances['utt-3'].issuer.idx == 'speaker-2'
        assert ds.utterances['utt-3'].start == 0
        assert ds.utterances['utt-3'].end == 1.5

        assert ds.utterances['utt-4'].idx == 'utt-4'
        assert ds.utterances['utt-4'].track.idx == 'file-3'
        assert ds.utterances['utt-4'].issuer.idx == 'speaker-2'
        assert ds.utterances['utt-4'].start == 1.5
        assert ds.utterances['utt-4'].end == 2.5

        assert ds.utterances['utt-5'].idx == 'utt-5'
        assert ds.utterances['utt-5'].track.idx == 'file-4'
        assert ds.utterances['utt-5'].issuer.idx == 'speaker-3'
        assert ds.utterances['utt-5'].start == 0
        assert ds.utterances['utt-5'].end == -1

    def test_load_issuers(self, reader, sample_corpus_path):
        ds = reader.load(sample_corpus_path)

        assert ds.num_issuers == 3

        assert ds.issuers['speaker-1'].idx == 'speaker-1'
        assert len(ds.issuers['speaker-1'].info) == 0
        assert type(ds.issuers['speaker-1']) == issuers.Speaker
        assert ds.issuers['speaker-1'].gender == issuers.Gender.MALE
        assert ds.issuers['speaker-1'].age_group == issuers.AgeGroup.ADULT
        assert ds.issuers['speaker-1'].native_language == 'deu'

        assert ds.issuers['speaker-2'].idx == 'speaker-2'
        assert len(ds.issuers['speaker-2'].info) == 0
        assert type(ds.issuers['speaker-2']) == issuers.Artist
        assert ds.issuers['speaker-2'].name == 'Ohooo'

        assert ds.issuers['speaker-3'].idx == 'speaker-3'
        assert len(ds.issuers['speaker-3'].info) == 1
        assert ds.issuers['speaker-3'].info['region'] == 'zh'

    def test_load_label_lists(self, reader, sample_corpus_path):
        ds = reader.load(sample_corpus_path)

        utt_1 = ds.utterances['utt-1']
        utt_3 = ds.utterances['utt-3']
        utt_4 = ds.utterances['utt-4']

        assert 'text' in utt_1.label_lists.keys()
        assert 'raw_text' in utt_3.label_lists.keys()

        assert len(utt_4.label_lists['text'].labels) == 3
        assert utt_4.label_lists['text'].labels[1].value == 'are'

        assert utt_4.label_lists['text'].labels[2].start == 3.5
        assert utt_4.label_lists['text'].labels[2].end == 4.2

    def test_load_label_meta(self, reader, sample_corpus_path):
        ds = reader.load(sample_corpus_path)

        utt_2 = ds.utterances['utt-2']
        utt_3 = ds.utterances['utt-3']
        utt_4 = ds.utterances['utt-4']

        assert len(utt_2.label_lists['text'].labels[0].meta) == 3
        assert utt_2.label_lists['text'].labels[0].meta['pron'] == 'huu'
        assert utt_2.label_lists['text'].labels[0].meta['duration'] == 2.3
        assert utt_2.label_lists['text'].labels[0].meta['stressed']

        assert len(utt_3.label_lists['text'].labels[0].meta) == 0

        assert len(utt_4.label_lists['text'].labels[2].meta) == 1
        assert utt_4.label_lists['text'].labels[2].meta['ex.'] == 19

    def test_load_features(self, reader, sample_corpus_path):
        ds = reader.load(sample_corpus_path)

        assert ds.feature_containers['mfcc'].path == os.path.join(sample_corpus_path, 'features', 'mfcc')
        assert ds.feature_containers['fbank'].path == os.path.join(sample_corpus_path, 'features', 'fbank')

    def test_load_subviews(self, reader, sample_corpus_path):
        ds = reader.load(sample_corpus_path)

        assert 'train' in ds.subviews.keys()
        assert 'dev' in ds.subviews.keys()

        assert len(ds.subviews['train'].filter_criteria) == 1
        assert len(ds.subviews['dev'].filter_criteria) == 1

        assert ds.subviews['train'].filter_criteria[0].utterance_idxs == {'utt-1', 'utt-2', 'utt-3'}
        assert ds.subviews['dev'].filter_criteria[0].utterance_idxs == {'utt-4', 'utt-5'}

        assert not ds.subviews['train'].filter_criteria[0].inverse
        assert not ds.subviews['dev'].filter_criteria[0].inverse


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

    def test_save_container_tracks(self, writer, reader, sample_corpus_path, tmpdir):
        # make sure relative path changes in contrast to self.ds.path
        out_path = os.path.join(tmpdir.strpath, 'somesubdir')
        os.makedirs(out_path)

        sample_corpus = reader.load(sample_corpus_path)

        writer.save(sample_corpus, out_path)

        with open(os.path.join(out_path, 'audio.txt'), 'r') as f:
            file_content = f.read()

        rel_path = os.path.relpath(os.path.join(sample_corpus_path, 'audio'), out_path)
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
