import unittest
import tempfile
import shutil
import os

import h5py

from pingu.corpus import preprocessing

from tests import resources


class OfflineProcessorDummy(preprocessing.OfflineProcessor):
    def process_sequence(self, frames, sampling_rate, utterance=None, corpus=None):
        return frames


class OfflineProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def test_process(self):
        ds = resources.create_dataset()
        processor = OfflineProcessorDummy(frame_size=4096, hop_size=2048)
        feat_path = os.path.join(self.tempdir, 'feats')

        processor.process_corpus(ds, feat_path)

        with h5py.File(feat_path, 'r') as f:
            utts = set(f.keys())

            assert utts == set(ds.utterances.keys())

            assert f['utt-1'].shape == (20, 4096)
            assert f['utt-2'].shape == (20, 4096)
            assert f['utt-3'].shape == (11, 4096)
            assert f['utt-4'].shape == (7, 4096)
            assert f['utt-5'].shape == (20, 4096)
