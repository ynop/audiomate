import os
import tempfile

from audiomate.utils import textfile


def test_read_separated_lines():
    file_path = os.path.join(os.path.dirname(__file__), 'multi_column_file.txt')

    expected = [
        ['a', '1', 'x'],
        ['b', '2', 'y'],
        ['c', '3', 'z']
    ]

    records = textfile.read_separated_lines(file_path, separator='\t')

    assert expected == records


def test_read_separated_lines_with_first_key():
    file_path = os.path.join(os.path.dirname(__file__), 'multi_column_file.txt')

    expected = {
        'a': ['1', 'x'],
        'b': ['2', 'y'],
        'c': ['3', 'z']
    }

    records = textfile.read_separated_lines_with_first_key(file_path, separator='\t')

    assert expected == records


def test_read_separated_keeping_empty_columns():
    file_path = os.path.join(os.path.dirname(__file__), 'empty_column_file.txt')

    expected = [
        ['a', '1', 'x', '', ''],
        ['b', '2', '', 'y'],
        ['c', '', '3', 'z'],
    ]

    records = textfile.read_separated_lines(file_path, separator='\t', keep_empty=True)

    assert expected == records


def test_read_key_value_lines():
    file_path = os.path.join(os.path.dirname(__file__), 'key_value_file.txt')

    expected = {
        'a': '1',
        'b': '2',
        'c': '3'
    }

    records = textfile.read_key_value_lines(file_path, separator=' ')

    assert expected == records


def test_write_separated_lines_sorted():
    data = {
        'hallo-0_103': 'hallo-0_1',
        'hallo-0_122': 'hallo-0',
        'hallo-0_1031': 'hallo-0_1',
        'hallo-0_1322': 'hallo-0',
        'hallo-0_1224': 'hallo-0'
    }

    f, path = tempfile.mkstemp(text=True)
    os.close(f)

    textfile.write_separated_lines(path, data, separator=' ', sort_by_column=1)

    f = open(path, 'r')
    value = f.read()
    f.close()

    lines = value.strip().split('\n')

    assert 5 == len(lines)

    assert lines[0].endswith('hallo-0')
    assert lines[1].endswith('hallo-0')
    assert lines[2].endswith('hallo-0')
    assert lines[3].endswith('hallo-0_1')
    assert lines[4].endswith('hallo-0_1')
