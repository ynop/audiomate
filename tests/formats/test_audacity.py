import os.path

from pingu.formats.audacity import read_label_file, write_label_file


class TestAudacityFormat(object):

    def test_read_label_file_en(self):
        path = os.path.join(os.path.dirname(__file__), 'audacity_labels_en.txt')
        labels = read_label_file(path)

        assert len(labels) == 2

        assert labels[0][0] == 43352.824046
        assert labels[0][1] == 43525.837661
        assert labels[0][2] == 'music'

        assert labels[1][0] == 43512.446969
        assert labels[1][1] == 43531.343483
        assert labels[1][2] == 'speech_male'

    def test_read_label_file_de(self):
        path = os.path.join(os.path.dirname(__file__), 'audacity_labels_de.txt')
        labels = read_label_file(path)

        assert len(labels) == 2

        assert labels[0][0] == 43352.824046
        assert labels[0][1] == 43525.837661
        assert labels[0][2] == 'music'

        assert labels[1][0] == 43512.446969
        assert labels[1][1] == 43531.343483
        assert labels[1][2] == 'speech_male'

    def test_write_label_file(self, tmpdir):
        path = os.path.join(tmpdir.strpath, 'audacity_labels.txt')
        entries = [
            [10.01, 11.08, 'music'],
            [11.08, 13.33, 'speech_male']
        ]

        write_label_file(path, entries)

        assert os.path.isfile(path)

        with open(path) as file:
            lines = file.readlines()

            assert len(lines) == 2

            assert lines[0] == '10.01\t11.08\tmusic\n'
            assert lines[1] == '11.08\t13.33\tspeech_male\n'
