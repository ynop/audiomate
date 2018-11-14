import os
import tempfile
import pytest

from audiomate import corpus
from audiomate.corpus import io
from audiomate.utils import textfile

from tests import resources


@pytest.fixture()
def writer():
    return io.MozillaDeepSpeechWriter()


@pytest.fixture()
def path():
    return tempfile.mkdtemp()


class TestMozillaDeepSpeechWriter:

    def test_save_all(self, writer, tmpdir):
        ds = resources.create_dataset()
        writer.save(ds, tmpdir.strpath)

        all_path = os.path.join(tmpdir.strpath, 'all.csv')

        assert os.path.isfile(all_path)

        records = textfile.read_separated_lines(all_path, separator=',')

        assert len(records) == 6

        # HEADER
        assert len(records[0]) == 3
        assert records[0][1] == 'wav_filesize'
        assert records[0][2] == 'transcript'

        # DATA RECORDS
        utts = {r[0]: (r[1], r[2]) for r in records[1:]}

        path = ds.utterances['utt-1'].track.path
        assert len(utts[path]) == 2
        assert utts[path][0] == '83090'
        assert utts[path][1] == ds.utterances['utt-1'].label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value

        path = ds.utterances['utt-2'].track.path
        assert len(utts[path]) == 2
        assert utts[path][0] == '83090'
        assert utts[path][1] == ds.utterances['utt-2'].label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value

        path = os.path.join(tmpdir.strpath, 'audio', 'utt-3.wav')
        assert len(utts[path]) == 2
        assert utts[path][0] == '48044'
        assert utts[path][1] == ds.utterances['utt-3'].label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value

        path = os.path.join(tmpdir.strpath, 'audio', 'utt-4.wav')
        assert len(utts[path]) == 2
        assert utts[path][0] == '32044'
        assert utts[path][1] == ds.utterances['utt-4'].label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value

        path = ds.utterances['utt-5'].track.path
        assert len(utts[path]) == 2
        assert utts[path][0] == '83090'
        assert utts[path][1] == ds.utterances['utt-5'].label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value

    def test_save_subset_train(self, writer, tmpdir):
        ds = resources.create_dataset()
        writer.save(ds, tmpdir.strpath)

        all_path = os.path.join(tmpdir.strpath, 'train.csv')

        assert os.path.isfile(all_path)

        records = textfile.read_separated_lines(all_path, separator=',')

        assert len(records) == 4

        # HEADER
        assert len(records[0]) == 3
        assert records[0][1] == 'wav_filesize'
        assert records[0][2] == 'transcript'

        # DATA RECORDS
        utts = {r[0]: (r[1], r[2]) for r in records[1:]}

        path = ds.utterances['utt-1'].track.path
        assert len(utts[path]) == 2
        assert utts[path][0] == '83090'
        assert utts[path][1] == ds.utterances['utt-1'].label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value

        path = ds.utterances['utt-2'].track.path
        assert len(utts[path]) == 2
        assert utts[path][0] == '83090'
        assert utts[path][1] == ds.utterances['utt-2'].label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value

        path = os.path.join(tmpdir.strpath, 'audio', 'utt-3.wav')
        assert len(utts[path]) == 2
        assert utts[path][0] == '48044'
        assert utts[path][1] == ds.utterances['utt-3'].label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value

    def test_save_subset_dev(self, writer, tmpdir):
        ds = resources.create_dataset()
        writer.save(ds, tmpdir.strpath)

        all_path = os.path.join(tmpdir.strpath, 'dev.csv')

        assert os.path.isfile(all_path)

        records = textfile.read_separated_lines(all_path, separator=',')

        assert len(records) == 3

        # HEADER
        assert len(records[0]) == 3
        assert records[0][1] == 'wav_filesize'
        assert records[0][2] == 'transcript'

        # DATA RECORDS
        utts = {r[0]: (r[1], r[2]) for r in records[1:]}

        path = os.path.join(tmpdir.strpath, 'audio', 'utt-4.wav')
        assert len(utts[path]) == 2
        assert utts[path][0] == '32044'
        assert utts[path][1] == ds.utterances['utt-4'].label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value

        path = ds.utterances['utt-5'].track.path
        assert len(utts[path]) == 2
        assert utts[path][0] == '83090'
        assert utts[path][1] == ds.utterances['utt-5'].label_lists[corpus.LL_WORD_TRANSCRIPT].labels[0].value
