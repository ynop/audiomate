import os.path

from audiomate import annotations
from audiomate.formats import audacity


class TestAudacityFormat:

    def test_read_label_file_en(self):
        path = os.path.join(os.path.dirname(__file__), 'audacity_labels_en.txt')
        labels = audacity.read_label_file(path)

        assert len(labels) == 2

        assert labels[0][0] == 43352.824046
        assert labels[0][1] == 43525.837661
        assert labels[0][2] == 'music'

        assert labels[1][0] == 43512.446969
        assert labels[1][1] == 43531.343483
        assert labels[1][2] == 'speech_male'

    def test_read_label_file_de(self):
        path = os.path.join(os.path.dirname(__file__), 'audacity_labels_de.txt')
        labels = audacity.read_label_file(path)

        assert len(labels) == 2

        assert labels[0][0] == 43352.824046
        assert labels[0][1] == 43525.837661
        assert labels[0][2] == 'music'

        assert labels[1][0] == 43512.446969
        assert labels[1][1] == 43531.343483
        assert labels[1][2] == 'speech_male'

    def test_read_label_file_with_empty_value(self):
        path = os.path.join(os.path.dirname(__file__), 'audacity_labels_empty_value.txt')
        labels = audacity.read_label_file(path)

        assert len(labels) == 3

        assert labels[0][0] == 1
        assert labels[0][1] == 4
        assert labels[0][2] == 'music'

        assert labels[1][0] == 4
        assert labels[1][1] == 7
        assert labels[1][2] == ''

        assert labels[2][0] == 7
        assert labels[2][1] == 9
        assert labels[2][2] == 'speech_male'

    def test_read_label_list_en(self):
        path = os.path.join(os.path.dirname(__file__), 'audacity_labels_en.txt')
        ll = audacity.read_label_list(path)

        assert ll == annotations.LabelList(labels=[
            annotations.Label('music', 43352.824046, 43525.837661),
            annotations.Label('speech_male', 43512.446969, 43531.343483),
        ])

    def test_read_label_list_de(self):
        path = os.path.join(os.path.dirname(__file__), 'audacity_labels_de.txt')
        ll = audacity.read_label_list(path)

        assert ll == annotations.LabelList(labels=[
            annotations.Label('music', 43352.824046, 43525.837661),
            annotations.Label('speech_male', 43512.446969, 43531.343483),
        ])

    def test_read_label_list_with_empty_value(self):
        path = os.path.join(os.path.dirname(__file__), 'audacity_labels_empty_value.txt')
        ll = audacity.read_label_list(path)

        assert ll == annotations.LabelList(labels=[
            annotations.Label('music', 1, 4),
            annotations.Label('', 4, 7),
            annotations.Label('speech_male', 7, 9),
        ])

    def test_write_label_file(self, tmpdir):
        path = os.path.join(tmpdir.strpath, 'audacity_labels.txt')
        entries = [
            [10.01, 11.08, 'music'],
            [11.08, 13.33, 'speech_male']
        ]

        audacity.write_label_file(path, entries)

        assert os.path.isfile(path)

        with open(path) as file:
            lines = file.readlines()

            assert len(lines) == 2

            assert lines[0] == '10.01\t11.08\tmusic\n'
            assert lines[1] == '11.08\t13.33\tspeech_male\n'

    def test_write_label_list(self, tmpdir):
        path = os.path.join(tmpdir.strpath, 'audacity_labels.txt')
        ll = annotations.LabelList(labels=[
            annotations.Label('music', 10.01, 11.08),
            annotations.Label('speech_male', 11.08, 13.33),
        ])

        audacity.write_label_list(path, ll)

        assert os.path.isfile(path)

        with open(path) as file:
            lines = file.readlines()

            assert len(lines) == 2

            assert lines[0] == '10.01\t11.08\tmusic\n'
            assert lines[1] == '11.08\t13.33\tspeech_male\n'
