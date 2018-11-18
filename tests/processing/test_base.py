import os

import numpy as np
import librosa
import h5py

import pytest

from audiomate import tracks
from audiomate import containers
from audiomate import processing

from tests import resources


class ProcessorDummy(processing.Processor):

    def __init__(self, frame_size_scale=1.0, hop_size_scale=1.0):
        self.called_with_data = []
        self.called_with_sr = []
        self.called_with_offset = []
        self.called_with_last = []
        self.called_with_utterance = []
        self.called_with_corpus = []

        self.mock_frame_size_scale = frame_size_scale
        self.mock_hop_size_scale = hop_size_scale

    def process_frames(self, data, sampling_rate, offset=0, last=False, utterance=None, corpus=None):
        self.called_with_data.append(data)
        self.called_with_sr.append(sampling_rate)
        self.called_with_offset.append(offset)
        self.called_with_last.append(last)
        self.called_with_utterance.append(utterance)
        self.called_with_corpus.append(corpus)

        return data

    def frame_transform(self, frame_size, hop_size):
        tf_frame_size = self.mock_frame_size_scale * frame_size
        tf_hop_size = self.mock_hop_size_scale * hop_size

        return tf_frame_size, tf_hop_size


@pytest.fixture()
def processor():
    return ProcessorDummy()


@pytest.fixture()
def sample_utterance():
    file_track = tracks.FileTrack('test_file', resources.sample_wav_file('wav_1.wav'))
    utterance = tracks.Utterance('test', file_track)
    return utterance


class TestProcessor:

    #
    # process_track
    #

    def test_process_track(self, processor, tmpdir):
        wav_path = os.path.join(tmpdir.strpath, 'file.wav')
        wav_content = np.random.random(22)

        librosa.output.write_wav(wav_path, wav_content, 4)
        file_track = tracks.FileTrack('idx', wav_path)

        processed = processor.process_track(file_track, frame_size=4, hop_size=2)

        assert processed.shape == (10, 4)
        assert processed.dtype == np.float32
        assert np.allclose(processed[0], wav_content[0:4], atol=0.0001)
        assert np.allclose(processed[9], wav_content[18:22], atol=0.0001)

        assert processor.called_with_sr == [4]
        assert processor.called_with_offset == [0]
        assert processor.called_with_last == [True]
        assert processor.called_with_utterance == [None]
        assert processor.called_with_corpus == [None]

    def test_process_track_smaller_than_frame_size(self, processor, tmpdir):
        wav_path = os.path.join(tmpdir.strpath, 'file.wav')
        wav_content = np.random.random(22)

        librosa.output.write_wav(wav_path, wav_content, 16000)
        file_track = tracks.FileTrack('idx', wav_path)

        processed = processor.process_track(file_track, frame_size=4096, hop_size=2048, sr=16000)

        assert processed.shape == (1, 4096)
        assert np.allclose(processed[0], np.pad(wav_content, (0, 4074), mode='constant'), atol=0.0001)

        assert processor.called_with_sr == [16000]
        assert processor.called_with_offset == [0]
        assert processor.called_with_last == [True]
        assert processor.called_with_utterance == [None]
        assert processor.called_with_corpus == [None]

    def test_process_empty_track_raises_error(self, processor, tmpdir):
        wav_path = os.path.join(tmpdir.strpath, 'file.wav')
        wav_content = np.random.random(0)

        librosa.output.write_wav(wav_path, wav_content, 16000)
        file_track = tracks.FileTrack('idx', wav_path)

        with pytest.raises(ValueError):
            processor.process_track(file_track, frame_size=4096, hop_size=2048, sr=16000)

    def test_process_track_with_downsampling(self, processor, tmpdir):
        wav_path = os.path.join(tmpdir.strpath, 'file.wav')
        wav_content = np.random.random(22)

        librosa.output.write_wav(wav_path, wav_content, 4)
        file_track = tracks.FileTrack('idx', wav_path)

        processed = processor.process_track(file_track, frame_size=4, hop_size=2, sr=2)

        assert processed.shape == (5, 4)

        assert processor.called_with_sr == [2]
        assert processor.called_with_offset == [0]
        assert processor.called_with_last == [True]
        assert processor.called_with_utterance == [None]
        assert processor.called_with_corpus == [None]

    #
    #   process_track_online
    #

    def test_process_track_online(self, processor, tmpdir):
        wav_path = os.path.join(tmpdir.strpath, 'file.wav')
        wav_content = np.random.random(174)

        librosa.output.write_wav(wav_path, wav_content, 16000)
        track = tracks.FileTrack('idx', wav_path)

        chunks = list(processor.process_track_online(track, frame_size=20, hop_size=10, chunk_size=8))

        assert len(chunks) == 3
        assert np.allclose(chunks[0][0], wav_content[0:20], atol=0.0001)
        assert np.allclose(chunks[2][-1], np.pad(wav_content[160:], (0, 6), mode='constant'), atol=0.0001)
        assert chunks[0].dtype == np.float32

        assert processor.called_with_sr == [16000, 16000, 16000]
        assert processor.called_with_offset == [0, 8, 16]
        assert processor.called_with_last == [False, False, True]
        assert processor.called_with_utterance == [None, None, None]
        assert processor.called_with_corpus == [None, None, None]

    def test_process_track_online_no_rest_frame(self, processor, tmpdir):
        wav_path = os.path.join(tmpdir.strpath, 'file.wav')
        wav_content = np.random.random(170)

        librosa.output.write_wav(wav_path, wav_content, 16000)
        track = tracks.FileTrack('idx', wav_path)

        chunks = list(processor.process_track_online(track, frame_size=20, hop_size=10, chunk_size=8))

        assert len(chunks) == 2
        assert np.allclose(chunks[0][0], wav_content[0:20], atol=0.0001)
        assert np.allclose(chunks[1][-1], wav_content[150:], atol=0.0001)

        assert processor.called_with_sr == [16000, 16000]
        assert processor.called_with_offset == [0, 8]
        assert processor.called_with_last == [False, True]
        assert processor.called_with_utterance == [None, None]
        assert processor.called_with_corpus == [None, None]

    #
    #   process_utterance_...
    #

    def test_process_utterance(self, processor, sample_utterance):
        data = processor.process_utterance(sample_utterance, frame_size=4096, hop_size=2048)

        assert data.shape == (20, 4096)

    def test_process_utterance_with_start_end(self, processor, sample_utterance):
        sample_utterance.start = 1.0
        sample_utterance.end = 1.5

        data = processor.process_utterance(sample_utterance, frame_size=4096, hop_size=2048)

        assert data.shape == (3, 4096)

    def test_process_utterance_online(self, processor, sample_utterance):
        chunks = list(processor.process_utterance_online(sample_utterance, frame_size=4096,
                                                         hop_size=2048, chunk_size=4))

        assert len(chunks) == 5
        assert np.vstack(chunks).shape == (20, 4096)

    def test_process_utterance_online_with_start_end(self, processor, sample_utterance):
        sample_utterance.start = 1.0
        sample_utterance.end = 1.5

        chunks = list(processor.process_utterance_online(sample_utterance, frame_size=4096,
                                                         hop_size=2048, chunk_size=4))

        assert len(chunks) == 1
        assert np.vstack(chunks).shape == (3, 4096)

    #
    #   process_corpus
    #

    def test_process_corpus(self, processor, tmpdir):
        ds = resources.create_dataset()
        feat_path = os.path.join(tmpdir.strpath, 'feats')

        processor.process_corpus(ds, feat_path, frame_size=4096, hop_size=2048)

        with h5py.File(feat_path, 'r') as f:
            utts = set(f.keys())

            assert utts == set(ds.utterances.keys())

            assert f['utt-1'].shape == (20, 4096)
            assert f['utt-2'].shape == (20, 4096)
            assert f['utt-3'].shape == (11, 4096)
            assert f['utt-4'].shape == (7, 4096)
            assert f['utt-5'].shape == (20, 4096)

    def test_process_corpus_with_downsampling(self, processor, tmpdir):
        ds = resources.create_dataset()
        feat_path = os.path.join(tmpdir.strpath, 'feats')

        processor.process_corpus(ds, feat_path, frame_size=4096, hop_size=2048, sr=8000)

        with h5py.File(feat_path, 'r') as f:
            utts = set(f.keys())

            assert utts == set(ds.utterances.keys())

            assert f['utt-1'].shape == (10, 4096)
            assert f['utt-2'].shape == (10, 4096)
            assert f['utt-3'].shape == (5, 4096)
            assert f['utt-4'].shape == (3, 4096)
            assert f['utt-5'].shape == (10, 4096)

    def test_process_corpus_sets_container_attributes(self, processor, tmpdir):
        ds = resources.create_dataset()
        feat_path = os.path.join(tmpdir.strpath, 'feats')

        feat_container = processor.process_corpus(ds, feat_path, frame_size=4096, hop_size=2048)

        with feat_container:
            assert feat_container.frame_size == 4096
            assert feat_container.hop_size == 2048
            assert feat_container.sampling_rate == 16000

    def test_process_corpus_sets_container_attributes_with_downsampling(self, processor, tmpdir):
        ds = resources.create_dataset()
        feat_path = os.path.join(tmpdir.strpath, 'feats')

        feat_container = processor.process_corpus(ds, feat_path, frame_size=4096, hop_size=2048, sr=8000)

        with feat_container:
            assert feat_container.frame_size == 4096
            assert feat_container.hop_size == 2048
            assert feat_container.sampling_rate == 8000

    def test_process_corpus_with_frame_hop_size_change_stores_correct(self, processor, tmpdir):
        ds = resources.create_dataset()
        feat_path = os.path.join(tmpdir.strpath, 'feats')

        processor.mock_frame_size_scale = 2.5
        processor.mock_hop_size_scale = 5
        processor.process_corpus(ds, feat_path, frame_size=4096, hop_size=2048)

        fc = containers.FeatureContainer(feat_path)
        fc.open()

        assert fc.frame_size == 10240
        assert fc.hop_size == 10240

        fc.close()

    #
    #   process_corpus_online
    #

    def test_process_corpus_online(self, processor, tmpdir):
        ds = resources.create_dataset()
        feat_path = os.path.join(tmpdir.strpath, 'feats')

        processor.process_corpus_online(ds, feat_path, frame_size=4096, hop_size=2048)

        with h5py.File(feat_path, 'r') as f:
            utts = set(f.keys())

            assert utts == set(ds.utterances.keys())

            assert f['utt-1'].shape == (20, 4096)
            assert f['utt-2'].shape == (20, 4096)
            assert f['utt-3'].shape == (11, 4096)
            assert f['utt-4'].shape == (7, 4096)
            assert f['utt-5'].shape == (20, 4096)

    def test_process_corpus_online_sets_container_attributes(self, processor, tmpdir):
        ds = resources.create_dataset()
        feat_path = os.path.join(tmpdir.strpath, 'feats')

        feat_container = processor.process_corpus_online(ds, feat_path, frame_size=4096, hop_size=2048)

        with feat_container:
            assert feat_container.frame_size == 4096
            assert feat_container.hop_size == 2048
            assert feat_container.sampling_rate == 16000

    def test_process_corpus_online_ignore_returning_none(self, processor, tmpdir):
        ds = resources.create_dataset()
        feat_path = os.path.join(tmpdir.strpath, 'feats')

        def return_none(*args, **kwargs):
            return None

        processor.process_frames = return_none
        processor.process_corpus_online(ds, feat_path, frame_size=4096, hop_size=2048)

        assert True

    def test_process_corpus_online_with_frame_hop_size_change_stores_correct(self, processor, tmpdir):
        ds = resources.create_dataset()
        feat_path = os.path.join(tmpdir.strpath, 'feats')

        processor.mock_frame_size_scale = 0.5
        processor.mock_hop_size_scale = 0.25
        processor.process_corpus_online(ds, feat_path, frame_size=4096, hop_size=2048)

        fc = containers.FeatureContainer(feat_path)
        fc.open()

        assert fc.frame_size == 2048
        assert fc.hop_size == 512

        fc.close()

    #
    #   process_features_...
    #

    def test_process_features(self, processor, tmpdir):
        ds = resources.create_dataset()

        in_feat_path = os.path.join(tmpdir.strpath, 'in_feats')
        out_feat_path = os.path.join(tmpdir.strpath, 'out_feats')

        in_feats = containers.FeatureContainer(in_feat_path)
        utt_feats = np.arange(30).reshape(5, 6)

        with in_feats:
            in_feats.sampling_rate = 16000
            in_feats.frame_size = 400
            in_feats.hop_size = 160

            for utt_idx in ds.utterances.keys():
                in_feats.set(utt_idx, utt_feats)

        processor.process_features(ds, in_feats, out_feat_path)

        out_feats = containers.FeatureContainer(out_feat_path)

        with out_feats:
            assert len(out_feats.keys()) == 5

            assert np.array_equal(out_feats.get('utt-1', mem_map=False), utt_feats)
            assert np.array_equal(out_feats.get('utt-2', mem_map=False), utt_feats)
            assert np.array_equal(out_feats.get('utt-3', mem_map=False), utt_feats)
            assert np.array_equal(out_feats.get('utt-4', mem_map=False), utt_feats)
            assert np.array_equal(out_feats.get('utt-5', mem_map=False), utt_feats)

    def test_process_features_online(self, processor, tmpdir):
        ds = resources.create_dataset()

        in_feat_path = os.path.join(tmpdir.strpath, 'in_feats')
        out_feat_path = os.path.join(tmpdir.strpath, 'out_feats')

        in_feats = containers.FeatureContainer(in_feat_path)
        utt_feats = np.arange(30).reshape(5, 6)

        with in_feats:
            in_feats.sampling_rate = 16000
            in_feats.frame_size = 400
            in_feats.hop_size = 160

            for utt_idx in ds.utterances.keys():
                in_feats.set(utt_idx, utt_feats)

        processor.process_features_online(ds, in_feats, out_feat_path)

        out_feats = containers.FeatureContainer(out_feat_path)

        with out_feats:
            assert len(out_feats.keys()) == 5

            assert np.array_equal(out_feats.get('utt-1', mem_map=False), utt_feats)
            assert np.array_equal(out_feats.get('utt-2', mem_map=False), utt_feats)
            assert np.array_equal(out_feats.get('utt-3', mem_map=False), utt_feats)
            assert np.array_equal(out_feats.get('utt-4', mem_map=False), utt_feats)
            assert np.array_equal(out_feats.get('utt-5', mem_map=False), utt_feats)

    def test_process_features_online_with_given_chunk_size(self, processor, tmpdir):
        ds = resources.create_dataset()

        in_feat_path = os.path.join(tmpdir.strpath, 'in_feats')
        out_feat_path = os.path.join(tmpdir.strpath, 'out_feats')

        in_feats = containers.FeatureContainer(in_feat_path)
        utt_feats = np.arange(90).reshape(15, 6)

        with in_feats:
            in_feats.sampling_rate = 16000
            in_feats.frame_size = 400
            in_feats.hop_size = 160

            for utt_idx in ds.utterances.keys():
                in_feats.set(utt_idx, utt_feats)

        processor.process_features_online(ds, in_feats, out_feat_path, chunk_size=4)

        out_feats = containers.FeatureContainer(out_feat_path)

        assert len(processor.called_with_data) == 4 * 5
        assert processor.called_with_data[0].shape == (4, 6)
        assert processor.called_with_data[3].shape == (3, 6)

        with out_feats:
            assert len(out_feats.keys()) == 5

            assert np.array_equal(out_feats.get('utt-1', mem_map=False), utt_feats)
            assert np.array_equal(out_feats.get('utt-2', mem_map=False), utt_feats)
            assert np.array_equal(out_feats.get('utt-3', mem_map=False), utt_feats)
            assert np.array_equal(out_feats.get('utt-4', mem_map=False), utt_feats)
            assert np.array_equal(out_feats.get('utt-5', mem_map=False), utt_feats)

    def test_process_features_online_ignores_none(self, processor, tmpdir):
        ds = resources.create_dataset()

        in_feat_path = os.path.join(tmpdir.strpath, 'in_feats')
        out_feat_path = os.path.join(tmpdir.strpath, 'out_feats')

        in_feats = containers.FeatureContainer(in_feat_path)
        utt_feats = np.arange(90).reshape(15, 6)

        with in_feats:
            in_feats.sampling_rate = 16000
            in_feats.frame_size = 400
            in_feats.hop_size = 160

            for utt_idx in ds.utterances.keys():
                in_feats.set(utt_idx, utt_feats)

        def return_none(*args, **kwargs):
            return None

        processor.process_frames = return_none
        processor.process_features_online(ds, in_feats, out_feat_path, chunk_size=4)

        assert True

    def test_process_features_with_frame_hop_size_change_stores_correct(self, processor, tmpdir):
        ds = resources.create_dataset()

        in_feat_path = os.path.join(tmpdir.strpath, 'in_feats')
        out_feat_path = os.path.join(tmpdir.strpath, 'out_feats')

        in_feats = containers.FeatureContainer(in_feat_path)
        utt_feats = np.arange(30).reshape(5, 6)

        with in_feats:
            in_feats.sampling_rate = 16000
            in_feats.frame_size = 400
            in_feats.hop_size = 160

            for utt_idx in ds.utterances.keys():
                in_feats.set(utt_idx, utt_feats)

        processor.mock_frame_size_scale = 2.0
        processor.mock_hop_size_scale = 2.0
        processor.process_features(ds, in_feats, out_feat_path)

        out_feats = containers.FeatureContainer(out_feat_path)

        with out_feats:
            assert out_feats.frame_size == 800
            assert out_feats.hop_size == 320

    def test_process_features_online_with_frame_hop_size_change_stores_correct(self, processor, tmpdir):
        ds = resources.create_dataset()

        in_feat_path = os.path.join(tmpdir.strpath, 'in_feats')
        out_feat_path = os.path.join(tmpdir.strpath, 'out_feats')

        in_feats = containers.FeatureContainer(in_feat_path)
        utt_feats = np.arange(30).reshape(5, 6)

        with in_feats:
            in_feats.sampling_rate = 16000
            in_feats.frame_size = 400
            in_feats.hop_size = 160

            for utt_idx in ds.utterances.keys():
                in_feats.set(utt_idx, utt_feats)

        processor.mock_frame_size_scale = 2.0
        processor.mock_hop_size_scale = 2.0
        processor.process_features_online(ds, in_feats, out_feat_path)

        out_feats = containers.FeatureContainer(out_feat_path)

        with out_feats:
            assert out_feats.frame_size == 800
            assert out_feats.hop_size == 320
