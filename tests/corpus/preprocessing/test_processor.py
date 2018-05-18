import unittest
import tempfile
import shutil
import os

import h5py
import numpy as np
import pytest

from audiomate.corpus import assets
from audiomate.corpus import preprocessing

from tests import resources


class OfflineProcessorDummy(preprocessing.OfflineProcessor):

    def __init__(self):
        self.called_with_sr = None

    def process_sequence(self, frames, sampling_rate, utterance=None, corpus=None):
        self.called_with_sr = sampling_rate
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

    def test_process_corpus_with_downsampling(self):
        ds = resources.create_dataset()
        processor = OfflineProcessorDummy()
        feat_path = os.path.join(self.tempdir, 'feats')

        processor.process_corpus(ds, feat_path, frame_size=4096, hop_size=2048, sr=8000)

        with h5py.File(feat_path, 'r') as f:
            utts = set(f.keys())

            assert utts == set(ds.utterances.keys())

            assert f['utt-1'].shape == (10, 4096)
            assert f['utt-2'].shape == (10, 4096)
            assert f['utt-3'].shape == (5, 4096)
            assert f['utt-4'].shape == (3, 4096)
            assert f['utt-5'].shape == (10, 4096)

    def test_process_corpus_from_feature_container(self):
        ds = resources.create_dataset()
        processor = OfflineProcessorDummy()

        in_feat_path = os.path.join(self.tempdir, 'in_feats')
        out_feat_path = os.path.join(self.tempdir, 'out_feats')

        in_feats = assets.FeatureContainer(in_feat_path)
        utt_feats = np.arange(30).reshape(5, 6)

        with in_feats:
            in_feats.sampling_rate = 16000
            in_feats.frame_size = 400
            in_feats.hop_size = 160

            for utt_idx in ds.utterances.keys():
                in_feats.set(utt_idx, utt_feats)

        processor.process_corpus_from_feature_container(ds, in_feats, out_feat_path)

        out_feats = assets.FeatureContainer(out_feat_path)

        with out_feats:
            assert len(out_feats.keys()) == 5

            assert np.array_equal(out_feats.get('utt-1', mem_map=False), utt_feats)
            assert np.array_equal(out_feats.get('utt-2', mem_map=False), utt_feats)
            assert np.array_equal(out_feats.get('utt-3', mem_map=False), utt_feats)
            assert np.array_equal(out_feats.get('utt-4', mem_map=False), utt_feats)
            assert np.array_equal(out_feats.get('utt-5', mem_map=False), utt_feats)

    def test_process_utterance(self):
        processor = OfflineProcessorDummy()
        feat_path = os.path.join(self.tempdir, 'feats')
        file = assets.File('test_file', resources.sample_wav_file('wav_1.wav'))
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

    def test_process_corpus_sets_container_attributes_with_downsampling(self):
        ds = resources.create_dataset()
        processor = OfflineProcessorDummy()
        feat_path = os.path.join(self.tempdir, 'feats')

        feat_container = processor.process_corpus(ds, feat_path, frame_size=4096, hop_size=2048, sr=8000)

        with feat_container:
            assert feat_container.frame_size == 4096
            assert feat_container.hop_size == 2048
            assert feat_container.sampling_rate == 8000

    def test_process_empty_utterance_raises_error(self):
        processor = OfflineProcessorDummy()
        feat_path = os.path.join(self.tempdir, 'feats')
        file = assets.File('test_file', resources.sample_wav_file('empty.wav'))
        utterance = assets.Utterance('test', file)
        feat_container = assets.FeatureContainer(feat_path)
        feat_container.open()

        with pytest.raises(ValueError):
            processor.process_utterance(utterance, feat_container, frame_size=4096, hop_size=2048)

        feat_container.close()

    def test_process_utterance_smaller_than_frame_size(self):
        processor = OfflineProcessorDummy()
        feat_path = os.path.join(self.tempdir, 'feats')
        file = assets.File('test_file', resources.sample_wav_file('wav_200_samples.wav'))
        utterance = assets.Utterance('test', file)
        feat_container = assets.FeatureContainer(feat_path)
        feat_container.open()

        processor.process_utterance(utterance, feat_container, frame_size=4096, hop_size=2048)

        assert 'test' in feat_container.keys()
        assert feat_container.get('test').shape == (1, 4096)

        feat_container.close()

    def test_process_sequence_is_called_with_correct_sampling_rate(self):
        processor = OfflineProcessorDummy()
        feat_path = os.path.join(self.tempdir, 'feats')
        file = assets.File('test_file', resources.sample_wav_file('wav_1.wav'))
        utterance = assets.Utterance('test', file)
        feat_container = assets.FeatureContainer(feat_path)
        feat_container.open()

        processor.process_utterance(utterance, feat_container, frame_size=4096, hop_size=2048, sr=8000)

        assert processor.called_with_sr == 8000
