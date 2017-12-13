import re

from pingu.utils import textfile

__TIME_JUNK_PATTERN = re.compile(r'[^0-9.\-]')


def write_label_file(path, entries):
    """
    Writes an audacity label file. Start and end times are in seconds.

    Args:
        path (str): Path to write the file to.
        entries (list): List with entries to write.

    Example::

        >>> data = [
        >>>     [0.0, 0.2, 'sie'],
        >>>     [0.2, 2.2, 'hallo']
        >>> ]
        >>>
        >>> write_label_file('/some/path/to/file.txt', data)
    """

    textfile.write_separated_lines(path, entries, separator='\t')


def read_label_file(path):
    """
    Read the labels from an audacity label file.

    Args:
        path (str): Path to the label file.

    Returns:
        list: List of labels (start [sec], end [sec], label)

    Example::

        >>> read_label_file('/path/to/label/file.txt')
        [
            [0.0, 0.2, 'sie'],
            [0.2, 2.2, 'hallo']
        ]
    """
    labels = []

    for record in textfile.read_separated_lines_generator(path, separator='\t', max_columns=3):
        labels.append([float(_clean_time(record[0])), float(_clean_time(record[1])), str(record[2])])

    return labels


def _clean_time(time_str):
    # According to https://en.wikipedia.org/wiki/Decimal_mark, only the comma or the dot are valid as a decimal
    # separator and both may only be used as a decimal separator.
    return re.sub(__TIME_JUNK_PATTERN, '', time_str.replace(',', '.'))
