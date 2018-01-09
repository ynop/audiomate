import unittest
import tempfile
import shutil
import os

import h5py
import pytest

from pingu.corpus import assets
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

    def test_process_corpus(self):
        ds = resources.create_dataset()
        processor = OfflineProcessorDummy()
        feat_path = os.path.join(self.tempdir, 'feats')

        processor.process_corpus(ds, feat_path, frame_size=4096, hop_size=2048)

        with h5py.File(feat_path, 'r') as f:
            utts = set(f.keys())

            assert utts == set(ds.utterances.keys())

            assert f['utt-1'].shape == (20, 4096)
            assert f['utt-2'].shape == (20, 4096)
            assert f['utt-3'].shape == (11, 4096)
            assert f['utt-4'].shape == (7, 4096)
            assert f['utt-5'].shape == (20, 4096)

    def test_process_utterance(self):
        processor = OfflineProcessorDummy()
        feat_path = os.path.join(self.tempdir, 'feats')
        file = assets.File('test_file', resources.get_wav_file_path('wav_1.wav'))
        utterance = assets.Utterance('test', file)
        feat_container = assets.FeatureContainer(feat_path)
        feat_container.open()

        processor.process_utterance(utterance, feat_container, frame_size=4096, hop_size=2048)

        assert 'test' in feat_container.keys()
        assert feat_container.get('test').shape == (20, 4096)

    def test_process_corpus_sets_container_attributes(self):
        ds = resources.create_dataset()
        processor = OfflineProcessorDummy()
        feat_path = os.path.join(self.tempdir, 'feats')

        feat_container = processor.process_corpus(ds, feat_path, frame_size=4096, hop_size=2048)

        with feat_container:
            assert feat_container.frame_size == 4096
            assert feat_container.hop_size == 2048
            assert feat_container.sampling_rate == 16000

    def test_process_empty_utterance_raises_error(self):
        processor = OfflineProcessorDummy()
        feat_path = os.path.join(self.tempdir, 'feats')
        file = assets.File('test_file', resources.get_wav_file_path('empty.wav'))
        utterance = assets.Utterance('test', file)
        feat_container = assets.FeatureContainer(feat_path)
        feat_container.open()

        with pytest.raises(ValueError):
            processor.process_utterance(utterance, feat_container, frame_size=4096, hop_size=2048)

        feat_container.close()

    def test_process_utterance_smaller_than_frame_size(self):
        processor = OfflineProcessorDummy()
        feat_path = os.path.join(self.tempdir, 'feats')
        file = assets.File('test_file', resources.get_wav_file_path('wav_200_samples.wav'))
        utterance = assets.Utterance('test', file)
        feat_container = assets.FeatureContainer(feat_path)
        feat_container.open()

        processor.process_utterance(utterance, feat_container, frame_size=4096, hop_size=2048)

        assert 'test' in feat_container.keys()
        assert feat_container.get('test').shape == (1, 4096)

        feat_container.close()
