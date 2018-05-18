import os.path

from audiomate.corpus.assets import Label, LabelList
from audiomate.formats.audacity import read_label_file, read_label_list, write_label_file, write_label_list


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

    def test_read_label_file_with_empty_value(self):
        path = os.path.join(os.path.dirname(__file__), 'audacity_labels_empty_value.txt')
        labels = read_label_file(path)

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
        ll = read_label_list(path)

        assert len(ll) == 2

        assert ll[0].start == 43352.824046
        assert ll[0].end == 43525.837661
        assert ll[0].value == 'music'

        assert ll[1].start == 43512.446969
        assert ll[1].end == 43531.343483
        assert ll[1].value == 'speech_male'

    def test_read_label_list_de(self):
        path = os.path.join(os.path.dirname(__file__), 'audacity_labels_de.txt')
        ll = read_label_list(path)

        assert len(ll) == 2

        assert ll[0].start == 43352.824046
        assert ll[0].end == 43525.837661
        assert ll[0].value == 'music'

        assert ll[1].start == 43512.446969
        assert ll[1].end == 43531.343483
        assert ll[1].value == 'speech_male'

    def test_read_label_list_with_empty_value(self):
        path = os.path.join(os.path.dirname(__file__), 'audacity_labels_empty_value.txt')
        ll = read_label_list(path)

        assert len(ll) == 3

        assert ll[0].start == 1
        assert ll[0].end == 4
        assert ll[0].value == 'music'

        assert ll[1].start == 4
        assert ll[1].end == 7
        assert ll[1].value == ''

        assert ll[2].start == 7
        assert ll[2].end == 9
        assert ll[2].value == 'speech_male'

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

    def test_write_label_list(self, tmpdir):
        path = os.path.join(tmpdir.strpath, 'audacity_labels.txt')
        ll = LabelList(labels=[
            Label('music', 10.01, 11.08),
            Label('speech_male', 11.08, 13.33),
        ])

        write_label_list(path, ll)

        assert os.path.isfile(path)

        with open(path) as file:
            lines = file.readlines()

            assert len(lines) == 2

            assert lines[0] == '10.01\t11.08\tmusic\n'
            assert lines[1] == '11.08\t13.33\tspeech_male\n'
