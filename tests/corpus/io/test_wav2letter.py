import os
import tempfile
import pytest

from audiomate import corpus
from audiomate.corpus import io
from audiomate.utils import textfile

from tests import resources


@pytest.fixture()
def writer():
    return io.Wav2LetterWriter()


@pytest.fixture()
def path():
    return tempfile.mkdtemp()


class TestWav2LetterWriter:

    def test_save_all(self, writer, tmpdir):
        ds = resources.create_dataset()
        writer.save(ds, tmpdir.strpath)

        all_path = os.path.join(tmpdir.strpath, 'all.lst')

        assert os.path.isfile(all_path)

        records = textfile.read_separated_lines(all_path, separator=' ', max_columns=4)

        assert len(records) == 5

        # DATA RECORDS
        utts = {r[0]: (r[1], r[2], r[3]) for r in records}

        utt_idx = 'utt-1'
        assert utts[utt_idx][0] == ds.utterances[utt_idx].track.path
        assert utts[utt_idx][1] == '41523'
        assert utts[utt_idx][2] == ds.utterances[utt_idx].label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value

        utt_idx = 'utt-2'
        assert utts[utt_idx][0] == ds.utterances[utt_idx].track.path
        assert utts[utt_idx][1] == '41523'
        assert utts[utt_idx][2] == ds.utterances[utt_idx].label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value

        utt_idx = 'utt-3'
        assert utts[utt_idx][0] == os.path.join(tmpdir.strpath, 'audio', '{}.wav'.format(utt_idx))
        assert utts[utt_idx][1] == '24000'
        assert utts[utt_idx][2] == ds.utterances[utt_idx].label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value

        utt_idx = 'utt-4'
        assert utts[utt_idx][0] == os.path.join(tmpdir.strpath, 'audio', '{}.wav'.format(utt_idx))
        assert utts[utt_idx][1] == '16000'
        assert utts[utt_idx][2] == ds.utterances[utt_idx].label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value

        utt_idx = 'utt-5'
        assert utts[utt_idx][0] == ds.utterances[utt_idx].track.path
        assert utts[utt_idx][1] == '41523'
        assert utts[utt_idx][2] == ds.utterances[utt_idx].label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value

    def test_save_subset_train(self, writer, tmpdir):
        ds = resources.create_dataset()
        writer.save(ds, tmpdir.strpath)

        all_path = os.path.join(tmpdir.strpath, 'train.lst')

        assert os.path.isfile(all_path)

        records = textfile.read_separated_lines(all_path, separator=' ', max_columns=4)

        assert len(records) == 3

        utts = {r[0]: (r[1], r[2], r[3]) for r in records}

        utt_idx = 'utt-1'
        assert utts[utt_idx][0] == ds.utterances[utt_idx].track.path
        assert utts[utt_idx][1] == '41523'
        assert utts[utt_idx][2] == ds.utterances[utt_idx].label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value

        utt_idx = 'utt-2'
        assert utts[utt_idx][0] == ds.utterances[utt_idx].track.path
        assert utts[utt_idx][1] == '41523'
        assert utts[utt_idx][2] == ds.utterances[utt_idx].label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value

        utt_idx = 'utt-3'
        assert utts[utt_idx][0] == os.path.join(tmpdir.strpath, 'audio', '{}.wav'.format(utt_idx))
        assert utts[utt_idx][1] == '24000'
        assert utts[utt_idx][2] == ds.utterances[utt_idx].label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value

    def test_save_subset_dev(self, writer, tmpdir):
        ds = resources.create_dataset()
        writer.save(ds, tmpdir.strpath)

        all_path = os.path.join(tmpdir.strpath, 'dev.lst')

        assert os.path.isfile(all_path)

        records = textfile.read_separated_lines(all_path, separator=' ', max_columns=4)

        assert len(records) == 2

        # DATA RECORDS
        utts = {r[0]: (r[1], r[2], r[3]) for r in records}

        utt_idx = 'utt-4'
        assert utts[utt_idx][0] == os.path.join(tmpdir.strpath, 'audio', '{}.wav'.format(utt_idx))
        assert utts[utt_idx][1] == '16000'
        assert utts[utt_idx][2] == ds.utterances[utt_idx].label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value

        utt_idx = 'utt-5'
        assert utts[utt_idx][0] == ds.utterances[utt_idx].track.path
        assert utts[utt_idx][1] == '41523'
        assert utts[utt_idx][2] == ds.utterances[utt_idx].label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value
