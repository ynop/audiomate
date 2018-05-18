import collections

from audiomate.utils import textfile


def write_file(path, entries):
    """
    Writes a ctm file.

    Args:
        path (str): Path to write the file to.
        entries (list): List with entries to write. (entries -> wave-file, channel, start (seconds),
                        duration (seconds), label)

    Example::

        >>> data = [
        >>>     ["wave-ab", '1', 0.0, 0.82, "duda"],
        >>>     ["wave-xy", '1', 0.82, 0.57, "Jacques"],
        >>> ]
        >>>
        >>> write_file('/path/to/file.txt', data)
    """

    textfile.write_separated_lines(path, entries, separator=' ')


def read_file(path):
    """
    Reads a ctm file.

    Args:
        path (str): Path to the file

    Returns:
        (dict): Dictionary with entries.

    Example::

        >>> read_file('/path/to/file.txt')
        {
            'wave-ab': [
                ['1', 0.00, 0.07, 'HI', 1],
                ['1', 0.09, 0.08, 'AH', 1]
            ],
            'wave-xy': [
                ['1', 0.00, 0.07, 'HI', 1],
                ['1', 0.09, 0.08, 'AH', 1]
            ]
        }
    """
    gen = textfile.read_separated_lines_generator(path, max_columns=6,
                                                  ignore_lines_starting_with=[';;'])

    utterances = collections.defaultdict(list)

    for record in gen:
        values = record[1:len(record)]

        for i in range(len(values)):
            if i == 1 or i == 2 or i == 4:
                values[i] = float(values[i])

        utterances[record[0]].append(values)

    return utterances
