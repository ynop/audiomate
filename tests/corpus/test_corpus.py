import os

import numpy as np
import pytest

import audiomate
from audiomate import tracks
from audiomate import containers
from audiomate import annotations
from audiomate import issuers
from audiomate.corpus.subset import subview
from audiomate.corpus.io import MusanReader, KaldiWriter
from audiomate.corpus.io import UnknownWriterException, UnknownReaderException

from .. import resources


@pytest.fixture
def corpus():
    corpus = audiomate.Corpus()

    ex_file = tracks.FileTrack('existing_file', '../any/path.wav')
    ex_issuer = issuers.Issuer('existing_issuer')
    ex_utterance = tracks.Utterance('existing_utt', ex_file, issuer=ex_issuer)

    corpus.tracks['existing_file'] = ex_file
    corpus.issuers['existing_issuer'] = ex_issuer
    corpus.utterances['existing_utt'] = ex_utterance

    return corpus


class TestCorpus:

    #
    # TRACK ADD
    #

    def test_new_file(self, corpus):
        corpus.new_file('../some/path.wav', 'fid')

        assert corpus.num_tracks == 2
        assert corpus.tracks['fid'].idx == 'fid'
        assert corpus.tracks['fid'].path == os.path.abspath(
            os.path.join(os.getcwd(), '../some/path.wav')
        )

    def test_new_file_duplicate_idx(self, corpus):
        corpus.new_file('../some/other/path.wav', 'existing_file')

        assert corpus.num_tracks == 2
        assert corpus.tracks['existing_file_1'].idx == 'existing_file_1'
        assert corpus.tracks['existing_file_1'].path == os.path.abspath(
            os.path.join(os.getcwd(), '../some/other/path.wav')
        )

    def test_new_file_copy_file(self, corpus, tmpdir):
        file_path = resources.sample_wav_file('wav_1.wav')

        corpus.path = tmpdir.strpath
        corpus.new_file(file_path, 'fid', copy_file=True)

        assert corpus.num_tracks == 2
        assert corpus.tracks['fid'].path == os.path.join(tmpdir.strpath, 'files', 'fid.wav')

    def test_import_tracks(self, corpus):
        importing_tracks = [
            tracks.FileTrack('a', '/some/path.wav'),
            tracks.FileTrack('b', '/some/other/path.wav'),
            tracks.FileTrack('existing_file', '/some/otherer/path.wav'),
        ]

        idx_mapping = corpus.import_tracks(importing_tracks)

        assert corpus.num_tracks == 4

        assert 'a' in corpus.tracks.keys()
        assert corpus.tracks['a'].path == '/some/path.wav'

        assert 'b' in corpus.tracks.keys()
        assert corpus.tracks['b'].path == '/some/other/path.wav'

        assert 'existing_file_1' in corpus.tracks.keys()
        assert corpus.tracks['existing_file_1'].path == '/some/otherer/path.wav'

        assert len(idx_mapping) == 3
        assert 'a' in idx_mapping['a'].idx
        assert 'b' in idx_mapping['b'].idx
        assert idx_mapping['existing_file'].idx == 'existing_file_1'

    #
    #   UTT ADD
    #

    def test_new_utterance(self, corpus):
        corpus.new_utterance('some_utt', 'existing_file',
                             issuer_idx='existing_issuer', start=0, end=20)

        assert corpus.num_utterances == 2
        assert corpus.utterances['some_utt'].idx == 'some_utt'
        assert corpus.utterances['some_utt'].track.idx == 'existing_file'
        assert corpus.utterances['some_utt'].issuer.idx == 'existing_issuer'
        assert corpus.utterances['some_utt'].start == 0
        assert corpus.utterances['some_utt'].end == 20

    def test_new_utterance_duplicate_idx(self, corpus):
        corpus.new_utterance('existing_utt', 'existing_file',
                             issuer_idx='existing_issuer', start=0, end=20)

        assert corpus.num_utterances == 2
        assert corpus.utterances['existing_utt_1'].idx == 'existing_utt_1'
        assert corpus.utterances['existing_utt_1'].track.idx == 'existing_file'
        assert corpus.utterances['existing_utt_1'].issuer.idx == 'existing_issuer'
        assert corpus.utterances['existing_utt_1'].start == 0
        assert corpus.utterances['existing_utt_1'].end == 20

    def test_new_utterance_value_error_if_track_unknown(self, corpus):
        with pytest.raises(ValueError):
            corpus.new_utterance('some_utt', 'some_file', issuer_idx='iid', start=0, end=20)

    def test_import_utterances(self, corpus):
        importing_utterances = [
            tracks.Utterance('a', corpus.tracks['existing_file'],
                             corpus.issuers['existing_issuer'], 0, 10),
            tracks.Utterance('b', corpus.tracks['existing_file'],
                             corpus.issuers['existing_issuer'], 10, 20),
            tracks.Utterance('existing_utt', corpus.tracks['existing_file'],
                             corpus.issuers['existing_issuer'], 20, 30)
        ]

        mapping = corpus.import_utterances(importing_utterances)

        assert corpus.num_utterances == 4
        assert 'a' in corpus.utterances.keys()
        assert 'b' in corpus.utterances.keys()
        assert 'existing_utt_1' in corpus.utterances.keys()

        assert len(mapping) == 3
        assert mapping['a'].idx == 'a'
        assert mapping['b'].idx == 'b'
        assert mapping['existing_utt'].idx == 'existing_utt_1'

    def test_import_utterance_no_track(self, corpus):
        importing_utterances = [
            tracks.Utterance('a', tracks.FileTrack('notexist', 'notexist'),
                             corpus.issuers['existing_issuer'], 0, 10)
        ]

        with pytest.raises(ValueError):
            corpus.import_utterances(importing_utterances)

    def test_import_utterance_no_issuer(self, corpus):
        importing_utterances = [
            tracks.Utterance('a', corpus.tracks['existing_file'],
                             issuers.Issuer('notexist'), 0, 10)
        ]

        with pytest.raises(ValueError):
            corpus.import_utterances(importing_utterances)

    #
    #   ISSUER ADD
    #

    def test_new_issuer(self, corpus):
        corpus.new_issuer('some_iss', info={'hallo': 'velo'})

        assert corpus.num_issuers == 2
        assert corpus.issuers['some_iss'].idx == 'some_iss'
        assert corpus.issuers['some_iss'].info['hallo'] == 'velo'

    def test_new_issuer_duplicate_idx(self, corpus):
        corpus.new_issuer('existing_issuer', info={'hallo': 'velo'})

        assert corpus.num_issuers == 2
        assert corpus.issuers['existing_issuer_1'].idx == 'existing_issuer_1'
        assert corpus.issuers['existing_issuer_1'].info['hallo'] == 'velo'

    def test_import_issuers(self, corpus):
        importing_issuers = [
            issuers.Issuer('a'),
            issuers.Issuer('b'),
            issuers.Issuer('existing_issuer')
        ]

        mapping = corpus.import_issuers(importing_issuers)

        assert corpus.num_issuers == 4
        assert 'a' in corpus.issuers.keys()
        assert 'b' in corpus.issuers.keys()
        assert 'existing_issuer_1' in corpus.issuers.keys()

        assert len(mapping) == 3
        assert mapping['a'].idx == 'a'
        assert mapping['b'].idx == 'b'
        assert mapping['existing_issuer'].idx == 'existing_issuer_1'

    #
    #   FEAT CONT ADD
    #

    def test_new_feature_container(self, corpus, tmpdir):
        corpus.path = tmpdir.strpath
        corpus.new_feature_container('mfcc')

        assert corpus.num_feature_containers == 1
        assert corpus.feature_containers['mfcc'].path == os.path.join(
            tmpdir.strpath, 'features', 'mfcc'
        )

    #
    #   SUBVIEW ADD
    #
    def test_import_subview(self, corpus):
        train_set = subview.Subview(None, filter_criteria=[
            subview.MatchingUtteranceIdxFilter(utterance_idxs={'existing_utt'})
        ])

        corpus.import_subview('train', train_set)

        assert corpus.num_subviews == 1
        assert corpus.subviews['train'] == train_set
        assert corpus.subviews['train'].corpus == corpus

    #
    #   CREATION
    #

    def test_from_corpus(self):
        original = resources.create_dataset()
        copy = audiomate.Corpus.from_corpus(original)

        assert copy.num_tracks == 4
        assert copy.num_issuers == 3
        assert copy.num_utterances == 5
        assert copy.num_subviews == 2
        assert copy.num_feature_containers == 2

        original.tracks['wav-1'].path = '/changed/path.wav'
        assert original.tracks['wav-1'].path != copy.tracks['wav-1'].path

    def test_from_corpus_only_utterances_and_tracks(self):
        ds = audiomate.Corpus()
        ds.new_file('/random/path', 'file_1')
        ds.new_file('/random/path2', 'file_2')
        ds.new_utterance('utt_1', 'file_1')
        ds.new_utterance('utt_2', 'file_2')

        copy = audiomate.Corpus.from_corpus(ds)

        assert copy.num_tracks == 2
        assert copy.num_utterances == 2
        assert copy.num_issuers == 0

    #
    #    CORPUS READING
    #

    def test_load_throws_exception_when_reader_unknown(self):
        corpus = audiomate.Corpus()

        with pytest.raises(UnknownReaderException):
            corpus.load(resources.sample_corpus_path('default'), reader='does_not_exist')

    def test_load_with_default_reader_when_reader_unspecified(self):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('default'))

        assert corpus.name == 'default'
        assert corpus.path == resources.sample_corpus_path('default')
        assert corpus.num_tracks == 6
        assert 'file-1' in corpus.tracks
        assert 'file-2' in corpus.tracks
        assert 'file-3' in corpus.tracks
        assert 'file-4' in corpus.tracks

    def test_load_with_custom_reader_specified_by_name(self):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('musan'), reader='musan')

        assert corpus.name == 'musan'
        assert corpus.path == resources.sample_corpus_path('musan')
        assert corpus.num_tracks == 5
        assert 'music-fma-0000' in corpus.tracks
        assert 'noise-free-sound-0000' in corpus.tracks
        assert 'noise-free-sound-0001' in corpus.tracks
        assert 'speech-librivox-0000' in corpus.tracks
        assert 'speech-librivox-0001' in corpus.tracks

    def test_load_with_custom_reader_specified_by_instance(self):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('musan'), reader=MusanReader())

        assert corpus.name == 'musan'
        assert corpus.path == resources.sample_corpus_path('musan')
        assert corpus.num_tracks == 5
        assert 'music-fma-0000' in corpus.tracks
        assert 'noise-free-sound-0000' in corpus.tracks
        assert 'noise-free-sound-0001' in corpus.tracks
        assert 'speech-librivox-0000' in corpus.tracks
        assert 'speech-librivox-0001' in corpus.tracks

    #
    #    CORPUS SAVING
    #

    def test_save_at_corpus_path_throws_exception_when_writer_does_not_exist(self, tmpdir):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('default'))

        assert corpus.name == 'default'
        assert corpus.path == resources.sample_corpus_path('default')
        assert corpus.num_tracks == 6

        tempdir_contents = os.listdir(tmpdir.strpath)
        assert len(tempdir_contents) == 0

        corpus.path = tmpdir.strpath
        with pytest.raises(UnknownWriterException):
            corpus.save(writer='does_not_exist')

        assert len(os.listdir(tmpdir.strpath)) == 0

    def test_save_at_corpus_path_with_default_writer_when_writer_unspecified(self, tmpdir):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('default'))

        assert corpus.name == 'default'
        assert corpus.path == resources.sample_corpus_path('default')
        assert corpus.num_tracks == 6

        tempdir_contents = os.listdir(tmpdir.strpath)
        assert len(tempdir_contents) == 0

        corpus.path = tmpdir.strpath
        corpus.save()

        tempdir_contents = os.listdir(tmpdir.strpath)
        assert len(tempdir_contents) == 10

        assert 'files.txt' in tempdir_contents
        assert 'issuers.json' in tempdir_contents
        assert 'labels_raw_text.txt' in tempdir_contents
        assert 'labels_text.txt' in tempdir_contents
        assert 'utt_issuers.txt' in tempdir_contents
        assert 'utterances.txt' in tempdir_contents

    def test_save_at_corpus_path_with_writer_specified_by_name(self, tmpdir):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('kaldi'), reader='kaldi')

        assert corpus.name == 'kaldi'
        assert corpus.path == resources.sample_corpus_path('kaldi')
        assert corpus.path != tmpdir.strpath
        assert corpus.num_tracks == 4

        tempdir_contents = os.listdir(tmpdir.strpath)
        assert len(tempdir_contents) == 0

        corpus.path = tmpdir.strpath
        corpus.save(writer='kaldi')

        assert corpus.path == tmpdir.strpath

        tempdir_contents = os.listdir(tmpdir.strpath)
        assert len(tempdir_contents) == 5

        assert 'segments' in tempdir_contents
        assert 'spk2gender' in tempdir_contents
        assert 'text' in tempdir_contents
        assert 'utt2spk' in tempdir_contents
        assert 'wav.scp' in tempdir_contents

    def test_save_at_corpus_path_with_writer_specified_by_instance(self, tmpdir):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('kaldi'), reader='kaldi')

        assert corpus.name == 'kaldi'
        assert corpus.path == resources.sample_corpus_path('kaldi')
        assert corpus.path != tmpdir.strpath
        assert corpus.num_tracks == 4

        tempdir_contents = os.listdir(tmpdir.strpath)
        assert len(tempdir_contents) == 0

        corpus.path = tmpdir.strpath
        corpus.save(writer=KaldiWriter())

        assert corpus.path == tmpdir.strpath

        tempdir_contents = os.listdir(tmpdir.strpath)
        assert len(tempdir_contents) == 5

        assert 'segments' in tempdir_contents
        assert 'spk2gender' in tempdir_contents
        assert 'text' in tempdir_contents
        assert 'utt2spk' in tempdir_contents
        assert 'wav.scp' in tempdir_contents

    def test_save_at_path_throws_exception_when_writer_does_not_exist(self, tmpdir):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('default'))

        assert corpus.name == 'default'
        assert corpus.path == resources.sample_corpus_path('default')
        assert corpus.num_tracks == 6

        tempdir_contents = os.listdir(tmpdir.strpath)
        assert len(tempdir_contents) == 0

        with pytest.raises(UnknownWriterException):
            corpus.save_at(tmpdir.strpath, writer='does_not_exist')

        assert len(os.listdir(tmpdir.strpath)) == 0

    def test_save_at_path_with_default_writer_when_writer_unspecified(self, tmpdir):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('default'))

        assert corpus.name == 'default'
        assert corpus.path == resources.sample_corpus_path('default')
        assert corpus.num_tracks == 6

        tempdir_contents = os.listdir(tmpdir.strpath)
        assert len(tempdir_contents) == 0

        corpus.save_at(tmpdir.strpath)

        tempdir_contents = os.listdir(tmpdir.strpath)
        assert len(tempdir_contents) == 10

        assert 'files.txt' in tempdir_contents
        assert 'issuers.json' in tempdir_contents
        assert 'labels_raw_text.txt' in tempdir_contents
        assert 'labels_text.txt' in tempdir_contents
        assert 'utt_issuers.txt' in tempdir_contents
        assert 'utterances.txt' in tempdir_contents

    def test_save_at_path_with_writer_specified_by_name(self, tmpdir):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('kaldi'), reader='kaldi')

        assert corpus.name == 'kaldi'
        assert corpus.path == resources.sample_corpus_path('kaldi')
        assert corpus.path != tmpdir.strpath
        assert corpus.num_tracks == 4

        tempdir_contents = os.listdir(tmpdir.strpath)
        assert len(tempdir_contents) == 0

        corpus.save_at(tmpdir.strpath, writer='kaldi')

        assert corpus.path == tmpdir.strpath

        tempdir_contents = os.listdir(tmpdir.strpath)
        assert len(tempdir_contents) == 5

        assert 'segments' in tempdir_contents
        assert 'spk2gender' in tempdir_contents
        assert 'text' in tempdir_contents
        assert 'utt2spk' in tempdir_contents
        assert 'wav.scp' in tempdir_contents

    def test_save_at_path_with_writer_specified_by_instance(self, tmpdir):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('kaldi'), reader='kaldi')

        assert corpus.name == 'kaldi'
        assert corpus.path == resources.sample_corpus_path('kaldi')
        assert corpus.path != tmpdir.strpath
        assert corpus.num_tracks == 4

        tempdir_contents = os.listdir(tmpdir.strpath)
        assert len(tempdir_contents) == 0

        corpus.save_at(tmpdir.strpath, writer=KaldiWriter())

        assert corpus.path == tmpdir.strpath

        tempdir_contents = os.listdir(tmpdir.strpath)
        assert len(tempdir_contents) == 5

        assert 'segments' in tempdir_contents
        assert 'spk2gender' in tempdir_contents
        assert 'text' in tempdir_contents
        assert 'utt2spk' in tempdir_contents
        assert 'wav.scp' in tempdir_contents

    def test_merge_corpus_tracks(self):
        main_corpus = resources.create_dataset()
        merging_corpus = resources.create_multi_label_corpus()

        main_corpus.merge_corpus(merging_corpus)

        assert main_corpus.num_tracks == 8

        assert set(main_corpus.tracks.keys()) == {
            'wav-1', 'wav_2', 'wav_3', 'wav_4',
            'wav-1_1', 'wav_2_1', 'wav_3_1', 'wav_4_1'
        }

        assert main_corpus.tracks['wav-1_1'].idx == 'wav-1_1'
        assert main_corpus.tracks['wav-1_1'].path == merging_corpus.tracks['wav-1'].path

    def test_merge_corpus_issuers(self):
        main_corpus = resources.create_dataset()
        merging_corpus = resources.create_multi_label_corpus()

        main_corpus.merge_corpus(merging_corpus)

        assert main_corpus.num_issuers == 6

        assert set(main_corpus.issuers.keys()) == {
            'spk-1', 'spk-2', 'spk-3',
            'spk-1_1', 'spk-2_1', 'spk-3_1'
        }

        assert main_corpus.issuers['spk-1_1'].idx == 'spk-1_1'
        assert main_corpus.issuers['spk-1_1'].info == merging_corpus.issuers['spk-1'].info
        assert len(main_corpus.issuers['spk-1_1'].utterances) == 2

    def test_merge_corpus_utterances(self):
        main_corpus = resources.create_dataset()
        merging_corpus = resources.create_multi_label_corpus()

        main_corpus.merge_corpus(merging_corpus)

        assert main_corpus.num_utterances == 13

        assert set(main_corpus.utterances.keys()) == {
            'utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5',
            'utt-1_1', 'utt-2_1', 'utt-3_1', 'utt-4_1',
            'utt-5_1', 'utt-6', 'utt-7', 'utt-8'
        }

        assert main_corpus.utterances['utt-2_1'].track == main_corpus.tracks['wav_2_1']
        assert main_corpus.utterances['utt-2_1'].issuer == main_corpus.issuers['spk-1_1']
        assert main_corpus.utterances['utt-2_1'].start == merging_corpus.utterances['utt-2'].start
        assert main_corpus.utterances['utt-2_1'].end == merging_corpus.utterances['utt-2'].end
        assert main_corpus.utterances['utt-2_1'] in main_corpus.issuers['spk-1_1'].utterances

    def test_merge_corpus_label_lists(self):
        main_corpus = resources.create_dataset()
        merging_corpus = resources.create_multi_label_corpus()

        main_corpus.merge_corpus(merging_corpus)

        assert set(main_corpus.utterances['utt-2_1'].label_lists.keys()) == {'default'}

        ll = main_corpus.utterances['utt-2_1'].label_lists['default']

        assert ll == annotations.LabelList(labels=[
            annotations.Label('music', 0, 5),
            annotations.Label('speech', 5, 12),
            annotations.Label('music', 13, 15)
        ])

    def test_merge_corpus_subviews(self):
        main_corpus = resources.create_dataset()
        merging_corpus = resources.create_multi_label_corpus()

        main_corpus.merge_corpus(merging_corpus)

        assert main_corpus.num_subviews == 4

        assert main_corpus.subviews.keys() == {'train', 'dev', 'train_1', 'dev_1'}
        assert main_corpus.subviews['train_1'].corpus == main_corpus
        assert set(main_corpus.subviews['train_1'].filter_criteria[0].utterance_idxs) == {
            'utt-4_1', 'utt-5_1', 'utt-6'
        }

    def test_merge_corpus_feature_containers(self):
        main_corpus = resources.create_dataset()
        merging_corpus = resources.create_multi_label_corpus()

        main_corpus.merge_corpus(merging_corpus)

        assert main_corpus.num_feature_containers == 4

        main_feats = main_corpus.feature_containers
        merge_feats = merging_corpus.feature_containers

        assert set(main_feats.keys()) == {'mfcc', 'mel', 'mfcc_1', 'energy'}
        assert main_feats['mfcc_1'].path == merge_feats['mfcc'].path
        assert main_feats['energy'].path == merge_feats['energy'].path

    def test_merge_corpora(self):
        ds1 = resources.create_dataset()
        ds2 = resources.create_multi_label_corpus()
        ds3 = resources.create_single_label_corpus()

        ds = audiomate.Corpus.merge_corpora([ds1, ds2, ds3])

        assert ds.num_tracks == 12
        assert ds.num_utterances == 21
        assert ds.num_issuers == 9
        assert ds.num_subviews == 4
        assert ds.num_feature_containers == 4

    #
    # Varia
    #

    def test_relocate_audio_to_single_container(self, tmpdir):
        corpus = audiomate.Corpus.load(resources.sample_corpus_path('default'))

        target_container_path = os.path.join(tmpdir.strpath, 'audio')
        corpus.relocate_audio_to_single_container(target_container_path)

        assert os.path.isfile(target_container_path)

        cont = containers.AudioContainer(target_container_path)
        cont.open()

        assert cont.keys() == [
            'file-1',
            'file-2',
            'file-3',
            'file-4',
            'file-5',
            'file-6',
        ]

        assert corpus.tracks['file-1'].idx == 'file-1'
        assert corpus.tracks['file-1'].container.path == cont.path

        assert corpus.utterances['utt-1'].track.idx == 'file-1'
        assert corpus.utterances['utt-1'].track.container.path == cont.path

        cont.close()

    def test_relocate_audio_to_wav_files(self, tmpdir):
        old_corpus = audiomate.Corpus.load(resources.sample_corpus_path('default'))
        new_corpus = audiomate.Corpus.from_corpus(old_corpus)

        target_path = os.path.join(tmpdir.strpath, 'audio')
        new_corpus.relocate_audio_to_wav_files(target_path)

        assert os.path.isdir(target_path)

        old_track = old_corpus.tracks['file-1']
        new_track = new_corpus.tracks['file-1']
        assert new_track.path == os.path.join(target_path, 'file-1.wav')
        assert np.allclose(
            old_track.read_samples(),
            new_track.read_samples()
        )

        old_track = old_corpus.tracks['file-5']
        new_track = new_corpus.tracks['file-5']
        assert new_track.path == os.path.join(target_path, 'file-5.wav')
        assert np.allclose(
            old_track.read_samples(),
            new_track.read_samples()
        )
