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
        self.corpus.utterances['existing_utt'] = assets.Utterance('existing_utt', 'existing_file', issuer_idx='existing_issuer')
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
        self.assertEqual(os.path.abspath(os.path.join(os.getcwd(), '../some/path.wav')), self.corpus.files['fid'].path)

    def test_new_file_duplicate_idx(self):
        self.corpus.new_file('../some/other/path.wav', 'existing_file')

        self.assertEqual(2, self.corpus.num_files)
        self.assertEqual('existing_file_1', self.corpus.files['existing_file_1'].idx)
        self.assertEqual(os.path.abspath(os.path.join(os.getcwd(), '../some/other/path.wav')), self.corpus.files['existing_file_1'].path)

    def test_new_file_copy_file(self):
        file_path, file_name = resources.dummy_wav_path_and_name()

        self.corpus.new_file(file_path, 'fid', copy_file=True)

        self.assertEqual(2, self.corpus.num_files)
        self.assertEqual(os.path.join(self.tempdir, 'files', 'fid.wav'), self.corpus.files['fid'].path)

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
        self.corpus.new_utterance('existing_utt', 'existing_file', issuer_idx='iid', start=0, end=20)

        self.assertEqual(2, self.corpus.num_utterances)
        self.assertEqual('existing_utt_1', self.corpus.utterances['existing_utt_1'].idx)
        self.assertEqual('existing_file', self.corpus.utterances['existing_utt_1'].file_idx)
        self.assertEqual('iid', self.corpus.utterances['existing_utt_1'].issuer_idx)
        self.assertEqual(0, self.corpus.utterances['existing_utt_1'].start)
        self.assertEqual(20, self.corpus.utterances['existing_utt_1'].end)

    def test_new_utterance_duplicate_idx(self):
        with self.assertRaises(ValueError):
            self.corpus.new_utterance('some_utt', 'some_file', issuer_idx='iid', start=0, end=20)

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

    #
    #   LABEL_LIST ADD
    #

    def test_new_label_list(self):
        self.corpus.new_label_list('existing_utt', labels=assets.Label('hallo'))

        self.assertEqual(1, len(self.corpus.label_lists['default']))
        self.assertEqual('hallo', self.corpus.label_lists['default']['existing_utt'][0].value)

    #
    #   FEAT CONT ADD
    #

    def test_new_feature_container(self):
        self.corpus.new_feature_container('mfcc')

        self.assertEqual(1, self.corpus.num_feature_containers)
        self.assertEqual(os.path.join(self.tempdir, 'features', 'mfcc'), self.corpus.feature_containers['mfcc'].path)
