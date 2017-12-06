import os
import shutil
import tempfile
import unittest

import pingu
from pingu.corpus import assets
from .. import resources


class CorpusTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.corpus = pingu.Corpus(self.tempdir)

        self.corpus.files['existing_file'] = assets.File('existing_file', '../any/path.wav')
        self.corpus.utterances['existing_utt'] = assets.Utterance('existing_utt', 'existing_file',
                                                                  issuer_idx='existing_issuer')
        self.corpus.issuers['existing_issuer'] = assets.Issuer('existing_issuer')

    def tearDown(self):
        shutil.rmtree(self.tempdir, ignore_errors=True)

    #
    # FILE ADD
    #

    def test_new_file(self):
        self.corpus.new_file('../some/path.wav', 'fid')

        self.assertEqual(2, self.corpus.num_files)
        self.assertEqual('fid', self.corpus.files['fid'].idx)
        self.assertEqual(os.path.abspath(os.path.join(os.getcwd(), '../some/path.wav')),
                         self.corpus.files['fid'].path)

    def test_new_file_duplicate_idx(self):
        self.corpus.new_file('../some/other/path.wav', 'existing_file')

        self.assertEqual(2, self.corpus.num_files)
        self.assertEqual('existing_file_1', self.corpus.files['existing_file_1'].idx)
        self.assertEqual(os.path.abspath(os.path.join(os.getcwd(), '../some/other/path.wav')),
                         self.corpus.files['existing_file_1'].path)

    def test_new_file_copy_file(self):
        file_path, file_name = resources.dummy_wav_path_and_name()

        self.corpus.new_file(file_path, 'fid', copy_file=True)

        self.assertEqual(2, self.corpus.num_files)
        self.assertEqual(os.path.join(self.tempdir, 'files', 'fid.wav'),
                         self.corpus.files['fid'].path)

    def test_import_files(self):
        importing_files = [
            assets.File('a', '/some/path.wav'),
            assets.File('b', '/some/other/path.wav'),
            assets.File('existing_file', '/some/otherer/path.wav'),
        ]

        idx_mapping = self.corpus.import_files(importing_files)

        self.assertEqual(4, self.corpus.num_files)

        self.assertIn('a', self.corpus.files.keys())
        self.assertEqual('/some/path.wav', self.corpus.files['a'].path)

        self.assertIn('b', self.corpus.files.keys())
        self.assertEqual('/some/other/path.wav', self.corpus.files['b'].path)

        self.assertIn('existing_file_1', self.corpus.files.keys())
        self.assertEqual('/some/otherer/path.wav', self.corpus.files['existing_file_1'].path)

        self.assertEqual(3, len(idx_mapping))
        self.assertEqual('a', idx_mapping['a'].idx)
        self.assertEqual('b', idx_mapping['b'].idx)
        self.assertEqual('existing_file_1', idx_mapping['existing_file'].idx)

    #
    #   UTT ADD
    #

    def test_new_utterance(self):
        self.corpus.new_utterance('some_utt', 'existing_file', issuer_idx='iid', start=0, end=20)

        self.assertEqual(2, self.corpus.num_utterances)
        self.assertEqual('some_utt', self.corpus.utterances['some_utt'].idx)
        self.assertEqual('existing_file', self.corpus.utterances['some_utt'].file_idx)
        self.assertEqual('iid', self.corpus.utterances['some_utt'].issuer_idx)
        self.assertEqual(0, self.corpus.utterances['some_utt'].start)
        self.assertEqual(20, self.corpus.utterances['some_utt'].end)

    def test_new_utterance_duplicate_idx(self):
        self.corpus.new_utterance('existing_utt', 'existing_file', issuer_idx='iid', start=0,
                                  end=20)

        self.assertEqual(2, self.corpus.num_utterances)
        self.assertEqual('existing_utt_1', self.corpus.utterances['existing_utt_1'].idx)
        self.assertEqual('existing_file', self.corpus.utterances['existing_utt_1'].file_idx)
        self.assertEqual('iid', self.corpus.utterances['existing_utt_1'].issuer_idx)
        self.assertEqual(0, self.corpus.utterances['existing_utt_1'].start)
        self.assertEqual(20, self.corpus.utterances['existing_utt_1'].end)

    def test_new_utterance_value_error_if_file_unknown(self):
        with self.assertRaises(ValueError):
            self.corpus.new_utterance('some_utt', 'some_file', issuer_idx='iid', start=0, end=20)

    def test_import_utterances(self):
        importing_utterances = [
            assets.Utterance('a', 'existing_file', 'existing_issuer', 0, 10),
            assets.Utterance('b', 'existing_file', 'existing_issuer', 10, 20),
            assets.Utterance('existing_utt', 'existing_file', 'existing_issuer', 20, 30)
        ]

        mapping = self.corpus.import_utterances(importing_utterances)

        self.assertEqual(4, self.corpus.num_utterances)
        self.assertIn('a', self.corpus.utterances.keys())
        self.assertIn('b', self.corpus.utterances.keys())
        self.assertIn('existing_utt_1', self.corpus.utterances.keys())

        self.assertEqual(3, len(mapping))
        self.assertEqual('a', mapping['a'].idx)
        self.assertEqual('b', mapping['b'].idx)
        self.assertEqual('existing_utt_1', mapping['existing_utt'].idx)

    def test_import_utterance_no_file(self):
        importing_utterances = [
            assets.Utterance('a', 'something_that_doesnt_exist', 'existing_issuer', 0, 10)
        ]

        with self.assertRaises(ValueError):
            self.corpus.import_utterances(importing_utterances)

    def test_import_utterance_no_speaker(self):
        importing_utterances = [
            assets.Utterance('a', 'existing_file', 'something_that_doesnt_exist', 0, 10)
        ]

        with self.assertRaises(ValueError):
            self.corpus.import_utterances(importing_utterances)

    #
    #   ISSUER ADD
    #

    def test_new_issuer(self):
        self.corpus.new_issuer('some_iss', info={'hallo': 'velo'})

        self.assertEqual(2, self.corpus.num_issuers)
        self.assertEqual('some_iss', self.corpus.issuers['some_iss'].idx)
        self.assertEqual('velo', self.corpus.issuers['some_iss'].info['hallo'])

    def test_new_issuer_duplicate_idx(self):
        self.corpus.new_issuer('existing_issuer', info={'hallo': 'velo'})

        self.assertEqual(2, self.corpus.num_issuers)
        self.assertEqual('existing_issuer_1', self.corpus.issuers['existing_issuer_1'].idx)
        self.assertEqual('velo', self.corpus.issuers['existing_issuer_1'].info['hallo'])

    def test_import_issuers(self):
        importing_issuers = [
            assets.Issuer('a'),
            assets.Issuer('b'),
            assets.Issuer('existing_issuer')
        ]

        mapping = self.corpus.import_issuers(importing_issuers)

        self.assertEqual(4, self.corpus.num_issuers)
        self.assertIn('a', self.corpus.issuers.keys())
        self.assertIn('b', self.corpus.issuers.keys())
        self.assertIn('existing_issuer_1', self.corpus.issuers.keys())

        self.assertEqual(3, len(mapping))
        self.assertEqual('a', mapping['a'].idx)
        self.assertEqual('b', mapping['b'].idx)
        self.assertEqual('existing_issuer_1', mapping['existing_issuer'].idx)

    #
    #   LABEL_LIST ADD
    #

    def test_new_label_list(self):
        self.corpus.new_label_list('existing_utt', labels=assets.Label('hallo'))

        self.assertEqual(1, len(self.corpus.label_lists['default']))
        self.assertEqual('hallo', self.corpus.label_lists['default']['existing_utt'][0].value)

    def test_import_label_list(self):
        ll = assets.LabelList('default', labels=[
            assets.Label('hello'),
            assets.Label('again')
        ])

        self.corpus.import_label_list('existing_utt', ll)

        self.assertEqual(1, len(self.corpus.label_lists['default']))
        self.assertIn('existing_utt', self.corpus.label_lists['default'].keys())

    #
    #   FEAT CONT ADD
    #

    def test_new_feature_container(self):
        self.corpus.new_feature_container('mfcc')

        self.assertEqual(1, self.corpus.num_feature_containers)
        self.assertEqual(os.path.join(self.tempdir, 'features', 'mfcc'),
                         self.corpus.feature_containers['mfcc'].path)

    #
    #   CREATION
    #

    def test_from_corpus(self):
        original = resources.create_dataset()
        copy = pingu.Corpus.from_corpus(original)

        self.assertEqual(4, copy.num_files)
        self.assertEqual(3, copy.num_issuers)
        self.assertEqual(5, copy.num_utterances)
        self.assertEqual(5, len(copy.label_lists['default']))

        original.files['wav-1'].path = '/changed/path.wav'
        self.assertNotEqual(original.files['wav-1'].path, copy.files['wav-1'].path)
