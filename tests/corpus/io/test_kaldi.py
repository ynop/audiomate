import os

import numpy as np

from audiomate import corpus
from audiomate import tracks
from audiomate import issuers
from audiomate.corpus import io
from audiomate.utils import textfile

import pytest

from tests import resources
from . import reader_test as rt


@pytest.fixture
def writer():
    return io.KaldiWriter()


class TestKaldiReader(rt.CorpusReaderTest):

    SAMPLE_PATH = resources.sample_corpus_path('kaldi')
    FILE_TRACK_BASE_PATH = os.path.join(SAMPLE_PATH, 'files')

    EXPECTED_NUMBER_OF_TRACKS = 4
    EXPECTED_TRACKS = [
        rt.ExpFileTrack('file-1', 'wav_1.wav'),
        rt.ExpFileTrack('file-2', 'wav_2.wav'),
        rt.ExpFileTrack('file-3', 'wav_3.wav'),
        rt.ExpFileTrack('file-4', 'wav_4.wav'),
    ]

    EXPECTED_NUMBER_OF_ISSUERS = 3
    EXPECTED_ISSUERS = [
        rt.ExpSpeaker('speaker-1', 2, issuers.Gender.MALE, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('speaker-2', 2, issuers.Gender.MALE, issuers.AgeGroup.UNKNOWN, None),
        rt.ExpSpeaker('speaker-3', 1, issuers.Gender.FEMALE, issuers.AgeGroup.UNKNOWN, None),
    ]

    EXPECTED_NUMBER_OF_UTTERANCES = 5
    EXPECTED_UTTERANCES = [
        rt.ExpUtterance('utt-1', 'file-1', 'speaker-1', 0, float('inf')),
        rt.ExpUtterance('utt-2', 'file-2', 'speaker-1', 0, float('inf')),
        rt.ExpUtterance('utt-3', 'file-3', 'speaker-2', 0, 15),
        rt.ExpUtterance('utt-4', 'file-3', 'speaker-2', 15, 25),
        rt.ExpUtterance('utt-5', 'file-4', 'speaker-3', 0, float('inf')),
    ]

    EXPECTED_LABEL_LISTS = {
        'utt-1': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
        ],
        'utt-2': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
        ],
        'utt-3': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
        ],
        'utt-4': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
        ],
        'utt-5': [
            rt.ExpLabelList(corpus.LL_WORD_TRANSCRIPT, 1),
        ],
    }

    EXPECTED_LABELS = {
        'utt-4': [
            rt.ExpLabel(corpus.LL_WORD_TRANSCRIPT, 'who are they', 0, float('inf')),
        ],
    }

    EXPECTED_NUMBER_OF_SUBVIEWS = 0

    def load(self):
        return io.KaldiReader().load(self.SAMPLE_PATH)


class TestKaldiWriter:

    def test_save(self, writer, tmpdir):
        ds = resources.create_dataset()
        path = tmpdir.strpath
        writer.save(ds, path)

        assert 'segments' in os.listdir(path)
        assert 'text' in os.listdir(path)
        assert 'utt2spk' in os.listdir(path)
        assert 'spk2gender' in os.listdir(path)
        assert 'wav.scp' in os.listdir(path)

    def test_write_wav_scp(self, writer, tmpdir):
        ds = resources.create_dataset()
        path = tmpdir.strpath
        writer.save(ds, path)

        content = textfile.read_separated_lines(
            os.path.join(path, 'wav.scp'),
            separator=' ',
            max_columns=2
        )

        wav_base = resources.get_resource_path(['wav_files'])
        wav_base = os.path.abspath(wav_base)

        assert content[0][0] == 'wav-1'
        assert content[0][1] == os.path.join(wav_base, 'wav_1.wav')
        assert content[1][0] == 'wav_2'
        assert content[1][1] == os.path.join(wav_base, 'wav_2.wav')
        assert content[2][0] == 'wav_3'
        assert content[2][1] == os.path.join(wav_base, 'wav_3.wav')
        assert content[3][0] == 'wav_4'
        assert content[3][1] == os.path.join(wav_base, 'wav_4.wav')

    def test_write_segments(self, writer, tmpdir):
        ds = resources.create_dataset()
        path = tmpdir.strpath
        writer.save(ds, path)

        content = textfile.read_separated_lines(
            os.path.join(path, 'segments'),
            separator=' ',
            max_columns=4
        )

        assert content[0][0] == 'utt-1'
        assert content[0][1] == 'wav-1'
        assert float(content[0][2]) == pytest.approx(0)
        assert float(content[0][3]) == pytest.approx(2.5951875)

        assert content[1][0] == 'utt-2'
        assert content[1][1] == 'wav_2'
        assert float(content[1][2]) == pytest.approx(0)
        assert float(content[1][3]) == pytest.approx(2.5951875)

        assert content[2][0] == 'utt-3'
        assert content[2][1] == 'wav_3'
        assert float(content[2][2]) == pytest.approx(0)
        assert float(content[2][3]) == pytest.approx(1.5)

        assert content[3][0] == 'utt-4'
        assert content[3][1] == 'wav_3'
        assert float(content[3][2]) == pytest.approx(1.5)
        assert float(content[3][3]) == pytest.approx(2.5)

        assert content[4][0] == 'utt-5'
        assert content[4][1] == 'wav_4'
        assert float(content[4][2]) == pytest.approx(0)
        assert float(content[4][3]) == pytest.approx(2.5951875)

    def test_exports_wavs_from_container_tracks(self, writer, tmpdir):
        path = tmpdir.strpath
        container_ds_path = os.path.join(path, 'container_ds')
        out_path = os.path.join(path, 'export')

        ds = resources.create_dataset()
        ds.relocate_audio_to_single_container(container_ds_path)

        writer.save(ds, out_path)

        print(os.listdir(out_path))

        track_path = os.path.join(out_path, 'audio', 'wav-1.wav')
        track = tracks.FileTrack(None, track_path)
        assert os.path.isfile(track_path)
        assert track.duration == pytest.approx(2.5951875)
        assert np.allclose(
            track.read_samples(),
            ds.tracks['wav-1'].read_samples(),
            atol=1e-05
        )

        track_path = os.path.join(out_path, 'audio', 'wav_2.wav')
        track = tracks.FileTrack(None, track_path)
        assert os.path.isfile(track_path)
        assert track.duration == pytest.approx(2.5951875)

        track_path = os.path.join(out_path, 'audio', 'wav_3.wav')
        track = tracks.FileTrack(None, track_path)
        assert os.path.isfile(track_path)
        assert track.duration == pytest.approx(2.5951875)

        track_path = os.path.join(out_path, 'audio', 'wav_4.wav')
        track = tracks.FileTrack(None, track_path)
        assert os.path.isfile(track_path)
        assert track.duration == pytest.approx(2.5951875)
