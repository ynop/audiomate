import os

from audiomate.formats import trn


class TestTrnFormat:

    def test_write(self, tmpdir):
        path = os.path.join(tmpdir.strpath, 'transcriptions.txt')
        entries = {
            'utt-2': 'other text',
            'utt-1': 'some text',
        }

        trn.write(path, entries)

        with open(path, 'r') as f:
            lines = f.readlines()

        assert lines[0] == 'some text (utt-1)\n'
        assert lines[1] == 'other text (utt-2)'

    def test_read(self):
        path = os.path.join(os.path.dirname(__file__), 'trn_transcriptions.txt')
        entries = trn.read(path)

        assert entries['utt-2'] == 'hallo du'
        assert entries['utt-3'] == 'wer ist da'
        assert entries['utt-5'] == 'wie wer (oder wo) ist das'
