import os
import unittest

from audiomate.corpus.io import SpeechCommandsReader

from tests import resources


class SpeechCommandsReaderTest(unittest.TestCase):
    def setUp(self):
        self.reader = SpeechCommandsReader()

        self.ds_path = resources.sample_corpus_path('speech_commands')

    def test_read_files(self):
        ds = self.reader.load(self.ds_path)

        assert ds.num_files == 13

        assert ds.files['0b77ee66_nohash_0_bed'].idx == '0b77ee66_nohash_0_bed'
        assert ds.files['0b77ee66_nohash_0_bed'].path == os.path.join(self.ds_path, 'bed', '0b77ee66_nohash_0.wav')
        assert ds.files['0b77ee66_nohash_1_bed'].idx == '0b77ee66_nohash_1_bed'
        assert ds.files['0b77ee66_nohash_1_bed'].path == os.path.join(self.ds_path, 'bed', '0b77ee66_nohash_1.wav')
        assert ds.files['0b77ee66_nohash_2_bed'].idx == '0b77ee66_nohash_2_bed'
        assert ds.files['0b77ee66_nohash_2_bed'].path == os.path.join(self.ds_path, 'bed', '0b77ee66_nohash_2.wav')
        assert ds.files['0bde966a_nohash_0_bed'].idx == '0bde966a_nohash_0_bed'
        assert ds.files['0bde966a_nohash_0_bed'].path == os.path.join(self.ds_path, 'bed', '0bde966a_nohash_0.wav')
        assert ds.files['0bde966a_nohash_1_bed'].idx == '0bde966a_nohash_1_bed'
        assert ds.files['0bde966a_nohash_1_bed'].path == os.path.join(self.ds_path, 'bed', '0bde966a_nohash_1.wav')
        assert ds.files['0c40e715_nohash_0_bed'].idx == '0c40e715_nohash_0_bed'
        assert ds.files['0c40e715_nohash_0_bed'].path == os.path.join(self.ds_path, 'bed', '0c40e715_nohash_0.wav')

        marvin_path = os.path.join(self.ds_path, 'marvin')
        assert ds.files['d5c41d6a_nohash_0_marvin'].idx == 'd5c41d6a_nohash_0_marvin'
        assert ds.files['d5c41d6a_nohash_0_marvin'].path == os.path.join(marvin_path, 'd5c41d6a_nohash_0.wav')
        assert ds.files['d7a58714_nohash_0_marvin'].idx == 'd7a58714_nohash_0_marvin'
        assert ds.files['d7a58714_nohash_0_marvin'].path == os.path.join(marvin_path, 'd7a58714_nohash_0.wav')
        assert ds.files['d8a5ace5_nohash_0_marvin'].idx == 'd8a5ace5_nohash_0_marvin'
        assert ds.files['d8a5ace5_nohash_0_marvin'].path == os.path.join(marvin_path, 'd8a5ace5_nohash_0.wav')

        assert ds.files['0a7c2a8d_nohash_0_one'].idx == '0a7c2a8d_nohash_0_one'
        assert ds.files['0a7c2a8d_nohash_0_one'].path == os.path.join(self.ds_path, 'one', '0a7c2a8d_nohash_0.wav')
        assert ds.files['0b77ee66_nohash_0_one'].idx == '0b77ee66_nohash_0_one'
        assert ds.files['0b77ee66_nohash_0_one'].path == os.path.join(self.ds_path, 'one', '0b77ee66_nohash_0.wav')
        assert ds.files['c1b7c224_nohash_0_one'].idx == 'c1b7c224_nohash_0_one'
        assert ds.files['c1b7c224_nohash_0_one'].path == os.path.join(self.ds_path, 'one', 'c1b7c224_nohash_0.wav')
        assert ds.files['c1b7c224_nohash_1_one'].idx == 'c1b7c224_nohash_1_one'
        assert ds.files['c1b7c224_nohash_1_one'].path == os.path.join(self.ds_path, 'one', 'c1b7c224_nohash_1.wav')

    def test_read_issuers(self):
        ds = self.reader.load(self.ds_path)

        assert ds.num_issuers == 8

        assert ds.issuers['0b77ee66'].idx == '0b77ee66'
        assert ds.issuers['0bde966a'].idx == '0bde966a'
        assert ds.issuers['0c40e715'].idx == '0c40e715'
        assert ds.issuers['d5c41d6a'].idx == 'd5c41d6a'
        assert ds.issuers['d7a58714'].idx == 'd7a58714'
        assert ds.issuers['d8a5ace5'].idx == 'd8a5ace5'
        assert ds.issuers['0a7c2a8d'].idx == '0a7c2a8d'
        assert ds.issuers['c1b7c224'].idx == 'c1b7c224'

    def test_read_utterances(self):
        ds = self.reader.load(self.ds_path)

        assert ds.num_utterances == 13

        assert ds.utterances['0b77ee66_nohash_0_bed'].idx == '0b77ee66_nohash_0_bed'
        assert ds.utterances['0b77ee66_nohash_0_bed'].file.idx == '0b77ee66_nohash_0_bed'
        assert ds.utterances['0b77ee66_nohash_0_bed'].start == 0
        assert ds.utterances['0b77ee66_nohash_0_bed'].end == -1
        assert ds.utterances['0b77ee66_nohash_1_bed'].idx == '0b77ee66_nohash_1_bed'
        assert ds.utterances['0b77ee66_nohash_1_bed'].file.idx == '0b77ee66_nohash_1_bed'
        assert ds.utterances['0b77ee66_nohash_1_bed'].start == 0
        assert ds.utterances['0b77ee66_nohash_1_bed'].end == -1
        assert ds.utterances['0b77ee66_nohash_2_bed'].idx == '0b77ee66_nohash_2_bed'
        assert ds.utterances['0b77ee66_nohash_2_bed'].file.idx == '0b77ee66_nohash_2_bed'
        assert ds.utterances['0b77ee66_nohash_2_bed'].start == 0
        assert ds.utterances['0b77ee66_nohash_2_bed'].end == -1
        assert ds.utterances['0bde966a_nohash_0_bed'].idx == '0bde966a_nohash_0_bed'
        assert ds.utterances['0bde966a_nohash_0_bed'].file.idx == '0bde966a_nohash_0_bed'
        assert ds.utterances['0bde966a_nohash_0_bed'].start == 0
        assert ds.utterances['0bde966a_nohash_0_bed'].end == -1
        assert ds.utterances['0bde966a_nohash_1_bed'].idx == '0bde966a_nohash_1_bed'
        assert ds.utterances['0bde966a_nohash_1_bed'].file.idx == '0bde966a_nohash_1_bed'
        assert ds.utterances['0bde966a_nohash_1_bed'].start == 0
        assert ds.utterances['0bde966a_nohash_1_bed'].end == -1
        assert ds.utterances['0c40e715_nohash_0_bed'].idx == '0c40e715_nohash_0_bed'
        assert ds.utterances['0c40e715_nohash_0_bed'].file.idx == '0c40e715_nohash_0_bed'
        assert ds.utterances['0c40e715_nohash_0_bed'].start == 0
        assert ds.utterances['0c40e715_nohash_0_bed'].end == -1

        assert ds.utterances['d5c41d6a_nohash_0_marvin'].idx == 'd5c41d6a_nohash_0_marvin'
        assert ds.utterances['d5c41d6a_nohash_0_marvin'].file.idx == 'd5c41d6a_nohash_0_marvin'
        assert ds.utterances['d5c41d6a_nohash_0_marvin'].start == 0
        assert ds.utterances['d5c41d6a_nohash_0_marvin'].end == -1
        assert ds.utterances['d7a58714_nohash_0_marvin'].idx == 'd7a58714_nohash_0_marvin'
        assert ds.utterances['d7a58714_nohash_0_marvin'].file.idx == 'd7a58714_nohash_0_marvin'
        assert ds.utterances['d7a58714_nohash_0_marvin'].start == 0
        assert ds.utterances['d7a58714_nohash_0_marvin'].end == -1
        assert ds.utterances['d8a5ace5_nohash_0_marvin'].idx == 'd8a5ace5_nohash_0_marvin'
        assert ds.utterances['d8a5ace5_nohash_0_marvin'].file.idx == 'd8a5ace5_nohash_0_marvin'
        assert ds.utterances['d8a5ace5_nohash_0_marvin'].start == 0
        assert ds.utterances['d8a5ace5_nohash_0_marvin'].end == -1

        assert ds.utterances['0a7c2a8d_nohash_0_one'].idx == '0a7c2a8d_nohash_0_one'
        assert ds.utterances['0a7c2a8d_nohash_0_one'].file.idx == '0a7c2a8d_nohash_0_one'
        assert ds.utterances['0a7c2a8d_nohash_0_one'].start == 0
        assert ds.utterances['0a7c2a8d_nohash_0_one'].end == -1
        assert ds.utterances['0b77ee66_nohash_0_one'].idx == '0b77ee66_nohash_0_one'
        assert ds.utterances['0b77ee66_nohash_0_one'].file.idx == '0b77ee66_nohash_0_one'
        assert ds.utterances['0b77ee66_nohash_0_one'].start == 0
        assert ds.utterances['0b77ee66_nohash_0_one'].end == -1
        assert ds.utterances['c1b7c224_nohash_0_one'].idx == 'c1b7c224_nohash_0_one'
        assert ds.utterances['c1b7c224_nohash_0_one'].file.idx == 'c1b7c224_nohash_0_one'
        assert ds.utterances['c1b7c224_nohash_0_one'].start == 0
        assert ds.utterances['c1b7c224_nohash_0_one'].end == -1
        assert ds.utterances['c1b7c224_nohash_1_one'].idx == 'c1b7c224_nohash_1_one'
        assert ds.utterances['c1b7c224_nohash_1_one'].file.idx == 'c1b7c224_nohash_1_one'
        assert ds.utterances['c1b7c224_nohash_1_one'].start == 0
        assert ds.utterances['c1b7c224_nohash_1_one'].end == -1

    def test_read_labels(self):
        ds = self.reader.load(self.ds_path)

        assert len(ds.utterances['0b77ee66_nohash_0_bed'].label_lists) == 1
        assert len(ds.utterances['0b77ee66_nohash_0_bed'].label_lists['default'].labels) == 1
        assert ds.utterances['0b77ee66_nohash_0_bed'].label_lists['default'].labels[0].value == 'bed'

        assert len(ds.utterances['0b77ee66_nohash_1_bed'].label_lists) == 1
        assert len(ds.utterances['0b77ee66_nohash_1_bed'].label_lists['default'].labels) == 1
        assert ds.utterances['0b77ee66_nohash_1_bed'].label_lists['default'].labels[0].value == 'bed'

        assert len(ds.utterances['0b77ee66_nohash_2_bed'].label_lists) == 1
        assert len(ds.utterances['0b77ee66_nohash_2_bed'].label_lists['default'].labels) == 1
        assert ds.utterances['0b77ee66_nohash_2_bed'].label_lists['default'].labels[0].value == 'bed'

        assert len(ds.utterances['0bde966a_nohash_0_bed'].label_lists) == 1
        assert len(ds.utterances['0bde966a_nohash_0_bed'].label_lists['default'].labels) == 1
        assert ds.utterances['0bde966a_nohash_0_bed'].label_lists['default'].labels[0].value == 'bed'

        assert len(ds.utterances['0bde966a_nohash_1_bed'].label_lists) == 1
        assert len(ds.utterances['0bde966a_nohash_1_bed'].label_lists['default'].labels) == 1
        assert ds.utterances['0bde966a_nohash_1_bed'].label_lists['default'].labels[0].value == 'bed'

        assert len(ds.utterances['0c40e715_nohash_0_bed'].label_lists) == 1
        assert len(ds.utterances['0c40e715_nohash_0_bed'].label_lists['default'].labels) == 1
        assert ds.utterances['0c40e715_nohash_0_bed'].label_lists['default'].labels[0].value == 'bed'

        assert len(ds.utterances['d5c41d6a_nohash_0_marvin'].label_lists) == 1
        assert len(ds.utterances['d5c41d6a_nohash_0_marvin'].label_lists['default'].labels) == 1
        assert ds.utterances['d5c41d6a_nohash_0_marvin'].label_lists['default'].labels[0].value == 'marvin'

        assert len(ds.utterances['d7a58714_nohash_0_marvin'].label_lists) == 1
        assert len(ds.utterances['d7a58714_nohash_0_marvin'].label_lists['default'].labels) == 1
        assert ds.utterances['d7a58714_nohash_0_marvin'].label_lists['default'].labels[0].value == 'marvin'

        assert len(ds.utterances['d8a5ace5_nohash_0_marvin'].label_lists) == 1
        assert len(ds.utterances['d8a5ace5_nohash_0_marvin'].label_lists['default'].labels) == 1
        assert ds.utterances['d8a5ace5_nohash_0_marvin'].label_lists['default'].labels[0].value == 'marvin'

        assert len(ds.utterances['0a7c2a8d_nohash_0_one'].label_lists) == 1
        assert len(ds.utterances['0a7c2a8d_nohash_0_one'].label_lists['default'].labels) == 1
        assert ds.utterances['0a7c2a8d_nohash_0_one'].label_lists['default'].labels[0].value == 'one'

        assert len(ds.utterances['0b77ee66_nohash_0_one'].label_lists) == 1
        assert len(ds.utterances['0b77ee66_nohash_0_one'].label_lists['default'].labels) == 1
        assert ds.utterances['0b77ee66_nohash_0_one'].label_lists['default'].labels[0].value == 'one'

        assert len(ds.utterances['c1b7c224_nohash_0_one'].label_lists) == 1
        assert len(ds.utterances['c1b7c224_nohash_0_one'].label_lists['default'].labels) == 1
        assert ds.utterances['c1b7c224_nohash_0_one'].label_lists['default'].labels[0].value == 'one'

        assert len(ds.utterances['c1b7c224_nohash_1_one'].label_lists) == 1
        assert len(ds.utterances['c1b7c224_nohash_1_one'].label_lists['default'].labels) == 1
        assert ds.utterances['c1b7c224_nohash_1_one'].label_lists['default'].labels[0].value == 'one'

    def test_read_subvies(self):
        ds = self.reader.load(self.ds_path)

        assert ds.num_subviews == 3

        assert ds.subviews['train'].num_utterances == 6
        assert '0b77ee66_nohash_0_bed' in ds.subviews['train'].utterances.keys()
        assert '0b77ee66_nohash_1_bed' in ds.subviews['train'].utterances.keys()
        assert '0b77ee66_nohash_2_bed' in ds.subviews['train'].utterances.keys()
        assert 'd5c41d6a_nohash_0_marvin' in ds.subviews['train'].utterances.keys()
        assert 'c1b7c224_nohash_0_one' in ds.subviews['train'].utterances.keys()
        assert 'c1b7c224_nohash_1_one' in ds.subviews['train'].utterances.keys()

        assert ds.subviews['dev'].num_utterances == 3
        assert '0c40e715_nohash_0_bed' in ds.subviews['dev'].utterances.keys()
        assert 'd8a5ace5_nohash_0_marvin' in ds.subviews['dev'].utterances.keys()
        assert '0a7c2a8d_nohash_0_one' in ds.subviews['dev'].utterances.keys()

        assert ds.subviews['test'].num_utterances == 4
        assert '0bde966a_nohash_0_bed' in ds.subviews['test'].utterances.keys()
        assert '0bde966a_nohash_1_bed' in ds.subviews['test'].utterances.keys()
        assert 'd7a58714_nohash_0_marvin' in ds.subviews['test'].utterances.keys()
        assert '0b77ee66_nohash_0_one' in ds.subviews['test'].utterances.keys()
