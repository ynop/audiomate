import os

import numpy as np

from audiomate import corpus
from audiomate import tracks
from audiomate import issuers
from audiomate.corpus import io
from audiomate.utils import textfile

import pytest

from tests import resources


@pytest.fixture
def reader():
    return io.KaldiReader()


@pytest.fixture
def writer():
    return io.KaldiWriter()


@pytest.fixture
def sample_path():
    return resources.sample_corpus_path('kaldi')


class TestKaldiReader:

    def test_load_tracks(self, reader, sample_path):
        ds = reader.load(sample_path)

        assert ds.num_tracks == 4

        assert ds.tracks['file-1'].idx == 'file-1'
        assert ds.tracks['file-1'].path == os.path.join(sample_path, 'files', 'wav_1.wav')
        assert ds.tracks['file-2'].idx == 'file-2'
        assert ds.tracks['file-2'].path == os.path.join(sample_path, 'files', 'wav_2.wav')
        assert ds.tracks['file-3'].idx == 'file-3'
        assert ds.tracks['file-3'].path == os.path.join(sample_path, 'files', 'wav_3.wav')
        assert ds.tracks['file-4'].idx == 'file-4'
        assert ds.tracks['file-4'].path == os.path.join(sample_path, 'files', 'wav_4.wav')

    def test_load_issuers(self, reader, sample_path):
        ds = reader.load(sample_path)

        assert ds.num_issuers == 3

        assert ds.issuers['speaker-1'].idx == 'speaker-1'
        assert type(ds.issuers['speaker-1']) == issuers.Speaker
        assert ds.issuers['speaker-1'].gender == issuers.Gender.MALE
        assert ds.issuers['speaker-1'].age_group == issuers.AgeGroup.UNKNOWN
        assert ds.issuers['speaker-1'].native_language is None

        assert ds.issuers['speaker-2'].idx == 'speaker-2'
        assert type(ds.issuers['speaker-2']) == issuers.Speaker
        assert ds.issuers['speaker-2'].gender == issuers.Gender.MALE
        assert ds.issuers['speaker-2'].age_group == issuers.AgeGroup.UNKNOWN
        assert ds.issuers['speaker-2'].native_language is None

        assert ds.issuers['speaker-3'].idx == 'speaker-3'
        assert type(ds.issuers['speaker-3']) == issuers.Speaker
        assert ds.issuers['speaker-3'].gender == issuers.Gender.FEMALE
        assert ds.issuers['speaker-3'].age_group == issuers.AgeGroup.UNKNOWN
        assert ds.issuers['speaker-3'].native_language is None

    def test_load_utterances(self, reader, sample_path):
        ds = reader.load(sample_path)

        assert ds.num_utterances == 5

        assert ds.utterances['utt-1'].idx == 'utt-1'
        assert ds.utterances['utt-1'].track.idx == 'file-1'
        assert ds.utterances['utt-1'].issuer.idx == 'speaker-1'
        assert ds.utterances['utt-1'].start == 0
        assert ds.utterances['utt-1'].end == -1

        assert ds.utterances['utt-2'].idx == 'utt-2'
        assert ds.utterances['utt-2'].track.idx == 'file-2'
        assert ds.utterances['utt-2'].issuer.idx == 'speaker-1'
        assert ds.utterances['utt-2'].start == 0
        assert ds.utterances['utt-2'].end == -1

        assert ds.utterances['utt-3'].idx == 'utt-3'
        assert ds.utterances['utt-3'].track.idx == 'file-3'
        assert ds.utterances['utt-3'].issuer.idx == 'speaker-2'
        assert ds.utterances['utt-3'].start == 0
        assert ds.utterances['utt-3'].end == 15

        assert ds.utterances['utt-4'].idx == 'utt-4'
        assert ds.utterances['utt-4'].track.idx == 'file-3'
        assert ds.utterances['utt-4'].issuer.idx == 'speaker-2'
        assert ds.utterances['utt-4'].start == 15
        assert ds.utterances['utt-4'].end == 25

        assert ds.utterances['utt-5'].idx == 'utt-5'
        assert ds.utterances['utt-5'].track.idx == 'file-4'
        assert ds.utterances['utt-5'].issuer.idx == 'speaker-3'
        assert ds.utterances['utt-5'].start == 0
        assert ds.utterances['utt-5'].end == -1

    def test_load_label_lists(self, reader, sample_path):
        ds = reader.load(sample_path)

        utt_1 = ds.utterances['utt-1']
        utt_2 = ds.utterances['utt-2']
        utt_3 = ds.utterances['utt-3']
        utt_4 = ds.utterances['utt-4']
        utt_5 = ds.utterances['utt-5']

        assert corpus.LL_WORD_TRANSCRIPT in utt_1.label_lists.keys()
        assert corpus.LL_WORD_TRANSCRIPT in utt_2.label_lists.keys()
        assert corpus.LL_WORD_TRANSCRIPT in utt_3.label_lists.keys()
        assert corpus.LL_WORD_TRANSCRIPT in utt_4.label_lists.keys()
        assert corpus.LL_WORD_TRANSCRIPT in utt_5.label_lists.keys()

        assert len(utt_4.label_lists[corpus.LL_WORD_TRANSCRIPT].labels) == 1
        assert utt_4.label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value == 'who are they'

        assert utt_4.label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].start == 0
        assert utt_4.label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].end == -1


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
