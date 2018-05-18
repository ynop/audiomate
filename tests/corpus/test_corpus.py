import os
import shutil
import tempfile
import unittest

import pytest

import audiomate
from audiomate.corpus import assets
from audiomate.corpus.subset import subview
from audiomate.corpus.io import MusanReader, KaldiWriter
from audiomate.corpus.io import UnknownWriterException, UnknownReaderException

from .. import resources


class CorpusTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.corpus = audiomate.Corpus(self.tempdir)

        self.ex_file = assets.File('existing_file', '../any/path.wav')
        self.ex_issuer = assets.Issuer('existing_issuer')
        self.ex_utterance = assets.Utterance('existing_utt', self.ex_file, issuer=self.ex_issuer)

        self.corpus.files['existing_file'] = self.ex_file
        self.corpus.issuers['existing_issuer'] = self.ex_issuer
        self.corpus.utterances['existing_utt'] = self.ex_utterance

    def tearDown(self):
        shutil.rmtree(self.tempdir, ignore_errors=True)

    #
    # FILE ADD
    #

    def test_new_file(self):
        self.corpus.new_file('../some/path.wav', 'fid')

        assert self.corpus.num_files == 2
        assert self.corpus.files['fid'].idx == 'fid'
        assert self.corpus.files['fid'].path == os.path.abspath(os.path.join(os.getcwd(), '../some/path.wav'))

    def test_new_file_duplicate_idx(self):
        self.corpus.new_file('../some/other/path.wav', 'existing_file')

        assert self.corpus.num_files == 2
        assert self.corpus.files['existing_file_1'].idx == 'existing_file_1'
        assert self.corpus.files['existing_file_1'].path == os.path.abspath(
            os.path.join(os.getcwd(), '../some/other/path.wav'))

    def test_new_file_copy_file(self):
        file_path = resources.sample_wav_file('wav_1.wav')

        self.corpus.new_file(file_path, 'fid', copy_file=True)

        assert self.corpus.num_files == 2
        assert self.corpus.files['fid'].path == os.path.join(self.tempdir, 'files', 'fid.wav')

    def test_import_files(self):
        importing_files = [
            assets.File('a', '/some/path.wav'),
            assets.File('b', '/some/other/path.wav'),
            assets.File('existing_file', '/some/otherer/path.wav'),
        ]

        idx_mapping = self.corpus.import_files(importing_files)

        assert self.corpus.num_files == 4

        assert 'a' in self.corpus.files.keys()
        assert self.corpus.files['a'].path == '/some/path.wav'

        assert 'b' in self.corpus.files.keys()
        assert self.corpus.files['b'].path == '/some/other/path.wav'

        assert 'existing_file_1' in self.corpus.files.keys()
        assert self.corpus.files['existing_file_1'].path == '/some/otherer/path.wav'

        assert len(idx_mapping) == 3
        assert 'a' in idx_mapping['a'].idx
        assert 'b' in idx_mapping['b'].idx
        assert idx_mapping['existing_file'].idx == 'existing_file_1'

    #
    #   UTT ADD
    #

    def test_new_utterance(self):
        self.corpus.new_utterance('some_utt', 'existing_file', issuer_idx='existing_issuer', start=0, end=20)

        assert self.corpus.num_utterances == 2
        assert self.corpus.utterances['some_utt'].idx == 'some_utt'
        assert self.corpus.utterances['some_utt'].file.idx == 'existing_file'
        assert self.corpus.utterances['some_utt'].issuer.idx == 'existing_issuer'
        assert self.corpus.utterances['some_utt'].start == 0
        assert self.corpus.utterances['some_utt'].end == 20

    def test_new_utterance_duplicate_idx(self):
        self.corpus.new_utterance('existing_utt', 'existing_file', issuer_idx='existing_issuer', start=0, end=20)

        assert self.corpus.num_utterances == 2
        assert self.corpus.utterances['existing_utt_1'].idx == 'existing_utt_1'
        assert self.corpus.utterances['existing_utt_1'].file.idx == 'existing_file'
        assert self.corpus.utterances['existing_utt_1'].issuer.idx == 'existing_issuer'
        assert self.corpus.utterances['existing_utt_1'].start == 0
        assert self.corpus.utterances['existing_utt_1'].end == 20

    def test_new_utterance_value_error_if_file_unknown(self):
        with pytest.raises(ValueError):
            self.corpus.new_utterance('some_utt', 'some_file', issuer_idx='iid', start=0, end=20)

    def test_import_utterances(self):
        importing_utterances = [
            assets.Utterance('a', self.ex_file, self.ex_issuer, 0, 10),
            assets.Utterance('b', self.ex_file, self.ex_issuer, 10, 20),
            assets.Utterance('existing_utt', self.ex_file, self.ex_issuer, 20, 30)
        ]

        mapping = self.corpus.import_utterances(importing_utterances)

        assert self.corpus.num_utterances == 4
        assert 'a' in self.corpus.utterances.keys()
        assert 'b' in self.corpus.utterances.keys()
        assert 'existing_utt_1' in self.corpus.utterances.keys()

        assert len(mapping) == 3
        assert mapping['a'].idx == 'a'
        assert mapping['b'].idx == 'b'
        assert mapping['existing_utt'].idx == 'existing_utt_1'

    def test_import_utterance_no_file(self):
        importing_utterances = [
            assets.Utterance('a', assets.File('notexist', 'notexist'), self.ex_issuer, 0, 10)
        ]

        with pytest.raises(ValueError):
            self.corpus.import_utterances(importing_utterances)

    def test_import_utterance_no_issuer(self):
        importing_utterances = [
            assets.Utterance('a', self.ex_file, assets.Issuer('notexist'), 0, 10)
        ]

        with pytest.raises(ValueError):
            self.corpus.import_utterances(importing_utterances)

    #
    #   ISSUER ADD
    #

    def test_new_issuer(self):
        self.corpus.new_issuer('some_iss', info={'hallo': 'velo'})

        assert self.corpus.num_issuers == 2
        assert self.corpus.issuers['some_iss'].idx == 'some_iss'
        assert self.corpus.issuers['some_iss'].info['hallo'] == 'velo'

    def test_new_issuer_duplicate_idx(self):
        self.corpus.new_issuer('existing_issuer', info={'hallo': 'velo'})

        assert self.corpus.num_issuers == 2
        assert self.corpus.issuers['existing_issuer_1'].idx == 'existing_issuer_1'
        assert self.corpus.issuers['existing_issuer_1'].info['hallo'] == 'velo'

    def test_import_issuers(self):
        importing_issuers = [
            assets.Issuer('a'),
            assets.Issuer('b'),
            assets.Issuer('existing_issuer')
        ]

        mapping = self.corpus.import_issuers(importing_issuers)

        assert self.corpus.num_issuers == 4
        assert 'a' in self.corpus.issuers.keys()
        assert 'b' in self.corpus.issuers.keys()
        assert 'existing_issuer_1' in self.corpus.issuers.keys()

        assert len(mapping) == 3
        assert mapping['a'].idx == 'a'
        assert mapping['b'].idx == 'b'
        assert mapping['existing_issuer'].idx == 'existing_issuer_1'

    #
    #   FEAT CONT ADD
    #

    def test_new_feature_container(self):
        self.corpus.new_feature_container('mfcc')

        assert self.corpus.num_feature_containers == 1
        assert self.corpus.feature_containers['mfcc'].path == os.path.join(self.tempdir, 'features', 'mfcc')

    #
    #   SUBVIEW ADD
    #
    def test_import_subview(self):
        train_set = subview.Subview(None, filter_criteria=[
            subview.MatchingUtteranceIdxFilter(utterance_idxs={'existing_utt'})
        ])

        self.corpus.import_subview('train', train_set)

        assert self.corpus.num_subviews == 1
        assert self.corpus.subviews['train'] == train_set
        assert self.corpus.subviews['train'].corpus == self.corpus

    #
    #   CREATION
    #

    def test_from_corpus(self):
        original = resources.create_dataset()
        copy = audiomate.Corpus.from_corpus(original)

        assert copy.num_files == 4
        assert copy.num_issuers == 3
        assert copy.num_utterances == 5
        assert copy.num_subviews == 2
        assert copy.num_feature_containers == 2

        original.files['wav-1'].path = '/changed/path.wav'
        assert original.files['wav-1'].path != copy.files['wav-1'].path

    def test_from_corpus_only_utterances_and_files(self):
        ds = audiomate.Corpus()
        ds.new_file('/random/path', 'file_1')
        ds.new_file('/random/path2', 'file_2')
        ds.new_utterance('utt_1', 'file_1')
        ds.new_utterance('utt_2', 'file_2')

        copy = audiomate.Corpus.from_corpus(ds)

        assert copy.num_files == 2
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
        assert corpus.num_files == 4
        assert 'file-1' in corpus.files
        assert 'file-2' in corpus.files
        assert 'file-3' in corpus.files
        assert 'file-4' in corpus.files

    def test_load_with_custom_reader_specified_by_name(self):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('musan'), reader='musan')

        assert corpus.name == 'musan'
        assert corpus.path == resources.sample_corpus_path('musan')
        assert corpus.num_files == 5
        assert 'music-fma-0000' in corpus.files
        assert 'noise-free-sound-0000' in corpus.files
        assert 'noise-free-sound-0001' in corpus.files
        assert 'speech-librivox-0000' in corpus.files
        assert 'speech-librivox-0001' in corpus.files

    def test_load_with_custom_reader_specified_by_instance(self):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('musan'), reader=MusanReader())

        assert corpus.name == 'musan'
        assert corpus.path == resources.sample_corpus_path('musan')
        assert corpus.num_files == 5
        assert 'music-fma-0000' in corpus.files
        assert 'noise-free-sound-0000' in corpus.files
        assert 'noise-free-sound-0001' in corpus.files
        assert 'speech-librivox-0000' in corpus.files
        assert 'speech-librivox-0001' in corpus.files

    #
    #    CORPUS SAVING
    #

    def test_save_at_corpus_path_throws_exception_when_writer_does_not_exist(self):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('default'))

        assert corpus.name == 'default'
        assert corpus.path == resources.sample_corpus_path('default')
        assert corpus.num_files == 4

        tempdir_contents = os.listdir(self.tempdir)
        assert len(tempdir_contents) == 0

        corpus.path = self.tempdir
        with pytest.raises(UnknownWriterException):
            corpus.save(writer='does_not_exist')

        assert len(os.listdir(self.tempdir)) == 0

    def test_save_at_corpus_path_with_default_writer_when_writer_unspecified(self):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('default'))

        assert corpus.name == 'default'
        assert corpus.path == resources.sample_corpus_path('default')
        assert corpus.num_files == 4

        tempdir_contents = os.listdir(self.tempdir)
        assert len(tempdir_contents) == 0

        corpus.path = self.tempdir
        corpus.save()

        tempdir_contents = os.listdir(self.tempdir)
        assert len(tempdir_contents) == 9

        assert 'files.txt' in tempdir_contents
        assert 'issuers.json' in tempdir_contents
        assert 'labels_raw_text.txt' in tempdir_contents
        assert 'labels_text.txt' in tempdir_contents
        assert 'utt_issuers.txt' in tempdir_contents
        assert 'utterances.txt' in tempdir_contents

    def test_save_at_corpus_path_with_writer_specified_by_name(self):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('kaldi'), reader='kaldi')

        assert corpus.name == 'kaldi'
        assert corpus.path == resources.sample_corpus_path('kaldi')
        assert corpus.path != self.tempdir
        assert corpus.num_files == 4

        tempdir_contents = os.listdir(self.tempdir)
        assert len(tempdir_contents) == 0

        corpus.path = self.tempdir
        corpus.save(writer='kaldi')

        assert corpus.path == self.tempdir

        tempdir_contents = os.listdir(self.tempdir)
        assert len(tempdir_contents) == 5

        assert 'segments' in tempdir_contents
        assert 'spk2gender' in tempdir_contents
        assert 'text' in tempdir_contents
        assert 'utt2spk' in tempdir_contents
        assert 'wav.scp' in tempdir_contents

    def test_save_at_corpus_path_with_writer_specified_by_instance(self):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('kaldi'), reader='kaldi')

        assert corpus.name == 'kaldi'
        assert corpus.path == resources.sample_corpus_path('kaldi')
        assert corpus.path != self.tempdir
        assert corpus.num_files == 4

        tempdir_contents = os.listdir(self.tempdir)
        assert len(tempdir_contents) == 0

        corpus.path = self.tempdir
        corpus.save(writer=KaldiWriter())

        assert corpus.path == self.tempdir

        tempdir_contents = os.listdir(self.tempdir)
        assert len(tempdir_contents) == 5

        assert 'segments' in tempdir_contents
        assert 'spk2gender' in tempdir_contents
        assert 'text' in tempdir_contents
        assert 'utt2spk' in tempdir_contents
        assert 'wav.scp' in tempdir_contents

    def test_save_at_path_throws_exception_when_writer_does_not_exist(self):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('default'))

        assert corpus.name == 'default'
        assert corpus.path == resources.sample_corpus_path('default')
        assert corpus.num_files == 4

        tempdir_contents = os.listdir(self.tempdir)
        assert len(tempdir_contents) == 0

        with pytest.raises(UnknownWriterException):
            corpus.save_at(self.tempdir, writer='does_not_exist')

        assert len(os.listdir(self.tempdir)) == 0

    def test_save_at_path_with_default_writer_when_writer_unspecified(self):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('default'))

        assert corpus.name == 'default'
        assert corpus.path == resources.sample_corpus_path('default')
        assert corpus.num_files == 4

        tempdir_contents = os.listdir(self.tempdir)
        assert len(tempdir_contents) == 0

        corpus.save_at(self.tempdir)

        tempdir_contents = os.listdir(self.tempdir)
        assert len(tempdir_contents) == 9

        assert 'files.txt' in tempdir_contents
        assert 'issuers.json' in tempdir_contents
        assert 'labels_raw_text.txt' in tempdir_contents
        assert 'labels_text.txt' in tempdir_contents
        assert 'utt_issuers.txt' in tempdir_contents
        assert 'utterances.txt' in tempdir_contents

    def test_save_at_path_with_writer_specified_by_name(self):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('kaldi'), reader='kaldi')

        assert corpus.name == 'kaldi'
        assert corpus.path == resources.sample_corpus_path('kaldi')
        assert corpus.path != self.tempdir
        assert corpus.num_files == 4

        tempdir_contents = os.listdir(self.tempdir)
        assert len(tempdir_contents) == 0

        corpus.save_at(self.tempdir, writer='kaldi')

        assert corpus.path == self.tempdir

        tempdir_contents = os.listdir(self.tempdir)
        assert len(tempdir_contents) == 5

        assert 'segments' in tempdir_contents
        assert 'spk2gender' in tempdir_contents
        assert 'text' in tempdir_contents
        assert 'utt2spk' in tempdir_contents
        assert 'wav.scp' in tempdir_contents

    def test_save_at_path_with_writer_specified_by_instance(self):
        corpus = audiomate.Corpus()
        corpus = corpus.load(resources.sample_corpus_path('kaldi'), reader='kaldi')

        assert corpus.name == 'kaldi'
        assert corpus.path == resources.sample_corpus_path('kaldi')
        assert corpus.path != self.tempdir
        assert corpus.num_files == 4

        tempdir_contents = os.listdir(self.tempdir)
        assert len(tempdir_contents) == 0

        corpus.save_at(self.tempdir, writer=KaldiWriter())

        assert corpus.path == self.tempdir

        tempdir_contents = os.listdir(self.tempdir)
        assert len(tempdir_contents) == 5

        assert 'segments' in tempdir_contents
        assert 'spk2gender' in tempdir_contents
        assert 'text' in tempdir_contents
        assert 'utt2spk' in tempdir_contents
        assert 'wav.scp' in tempdir_contents

    def test_merge_corpus_files(self):
        main_corpus = resources.create_dataset()
        merging_corpus = resources.create_multi_label_corpus()

        main_corpus.merge_corpus(merging_corpus)

        assert main_corpus.num_files == 8

        assert set(main_corpus.files.keys()) == {'wav-1', 'wav_2', 'wav_3', 'wav_4',
                                                 'wav-1_1', 'wav_2_1', 'wav_3_1', 'wav_4_1'}

        assert main_corpus.files['wav-1_1'].idx == 'wav-1_1'
        assert main_corpus.files['wav-1_1'].path == merging_corpus.files['wav-1'].path

    def test_merge_corpus_issuers(self):
        main_corpus = resources.create_dataset()
        merging_corpus = resources.create_multi_label_corpus()

        main_corpus.merge_corpus(merging_corpus)

        assert main_corpus.num_issuers == 6

        assert set(main_corpus.issuers.keys()) == {'spk-1', 'spk-2', 'spk-3',
                                                   'spk-1_1', 'spk-2_1', 'spk-3_1'}

        assert main_corpus.issuers['spk-1_1'].idx == 'spk-1_1'
        assert main_corpus.issuers['spk-1_1'].info == merging_corpus.issuers['spk-1'].info
        assert len(main_corpus.issuers['spk-1_1'].utterances) == 2

    def test_merge_corpus_utterances(self):
        main_corpus = resources.create_dataset()
        merging_corpus = resources.create_multi_label_corpus()

        main_corpus.merge_corpus(merging_corpus)

        assert main_corpus.num_utterances == 13

        assert set(main_corpus.utterances.keys()) == {'utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5',
                                                      'utt-1_1', 'utt-2_1', 'utt-3_1', 'utt-4_1', 'utt-5_1',
                                                      'utt-6', 'utt-7', 'utt-8'}

        assert main_corpus.utterances['utt-2_1'].file == main_corpus.files['wav_2_1']
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

        assert len(ll) == 3
        assert ll.labels[1].value == 'speech'
        assert ll.labels[1].start == 5
        assert ll.labels[1].end == 12
        assert ll.labels[1].meta == {}
        assert ll.labels[1].label_list == ll

    def test_merge_corpus_subviews(self):
        main_corpus = resources.create_dataset()
        merging_corpus = resources.create_multi_label_corpus()

        main_corpus.merge_corpus(merging_corpus)

        assert main_corpus.num_subviews == 4

        assert main_corpus.subviews.keys() == {'train', 'dev', 'train_1', 'dev_1'}
        assert main_corpus.subviews['train_1'].corpus == main_corpus
        assert set(main_corpus.subviews['train_1'].filter_criteria[0].utterance_idxs) == {'utt-4_1', 'utt-5_1', 'utt-6'}

    def test_merge_corpus_feature_containers(self):
        main_corpus = resources.create_dataset()
        merging_corpus = resources.create_multi_label_corpus()

        main_corpus.merge_corpus(merging_corpus)

        assert main_corpus.num_feature_containers == 4

        assert set(main_corpus.feature_containers.keys()) == {'mfcc', 'mel', 'mfcc_1', 'energy'}
        assert main_corpus.feature_containers['mfcc_1'].path == merging_corpus.feature_containers['mfcc'].path
        assert main_corpus.feature_containers['energy'].path == merging_corpus.feature_containers['energy'].path

    def test_merge_corpora(self):
        ds1 = resources.create_dataset()
        ds2 = resources.create_multi_label_corpus()
        ds3 = resources.create_single_label_corpus()

        ds = audiomate.Corpus.merge_corpora([ds1, ds2, ds3])

        assert ds.num_files == 12
        assert ds.num_utterances == 21
        assert ds.num_issuers == 9
        assert ds.num_subviews == 4
        assert ds.num_feature_containers == 4
