import os

import numpy as np
import librosa

from audiomate.utils import audio


def test_read_blocks(tmpdir):
    wav_path = os.path.join(tmpdir.strpath, 'file.wav')
    wav_content = np.random.random(10000)
    librosa.output.write_wav(wav_path, wav_content, 16000)

    data = [x for x in audio.read_blocks(wav_path, buffer_size=1000)]
    blocks = [x[0] for x in data]
    sr = [x[1] for x in data]

    assert np.allclose(np.concatenate(blocks), wav_content, atol=0.0001)
    assert np.concatenate(blocks).dtype == np.float32
    assert sr == [16000] * len(data)


def test_read_blocks_with_resampling(tmpdir):
    wav_path = os.path.join(tmpdir.strpath, 'file.wav')
    wav_content = np.random.random(10000)
    librosa.output.write_wav(wav_path, wav_content, 16000)

    data = [x for x in audio.read_blocks(wav_path, sr_target=8000, buffer_size=1000)]
    blocks = [x[0] for x in data]
    sr = [x[1] for x in data]

    assert np.concatenate(blocks).size == 5000
    assert np.concatenate(blocks).dtype == np.float32
    assert sr == [8000] * len(data)


def test_read_blocks_with_start_end(tmpdir):
    wav_path = os.path.join(tmpdir.strpath, 'file.wav')
    wav_content = np.random.random(10000)
    librosa.output.write_wav(wav_path, wav_content, 16000)

    blocks = [x[0] for x in audio.read_blocks(wav_path, start=0.1, end=0.3, buffer_size=1000)]

    assert np.concatenate(blocks).dtype == np.float32
    assert np.allclose(np.concatenate(blocks), wav_content[1600:4800], atol=0.0001)


def test_read_frames(tmpdir):
    wav_path = os.path.join(tmpdir.strpath, 'file.wav')
    wav_content = np.random.random(10044)
    librosa.output.write_wav(wav_path, wav_content, 16000)

    data = list(audio.read_frames(wav_path, frame_size=400, hop_size=160))
    frames = np.array([x[0] for x in data])
    sr = [x[1] for x in data]
    last = [x[2] for x in data]

    assert frames.shape == (62, 400)
    assert frames.dtype == np.float32
    assert np.allclose(frames[0], wav_content[:400], atol=0.0001)
    assert np.allclose(frames[61], np.pad(wav_content[9760:], (0, 116), mode='constant'), atol=0.0001)

    assert sr == [16000] * len(data)
    assert last[:-1] == [False] * (len(data) - 1)
    assert last[-1]


def test_read_frames_matches_length(tmpdir):
    wav_path = os.path.join(tmpdir.strpath, 'file.wav')
    wav_content = np.random.random(10000)
    librosa.output.write_wav(wav_path, wav_content, 16000)

    data = list(audio.read_frames(wav_path, frame_size=400, hop_size=160))
    frames = np.array([x[0] for x in data])
    sr = [x[1] for x in data]
    last = [x[2] for x in data]

    assert frames.shape == (61, 400)
    assert frames.dtype == np.float32
    assert np.allclose(frames[0], wav_content[:400], atol=0.0001)
    assert np.allclose(frames[60], wav_content[9600:], atol=0.0001)

    assert sr == [16000] * len(data)
    assert last[:-1] == [False] * (len(data) - 1)
    assert last[-1]
