"""
Functions for reading/writing sclite transcription files.

Description of the format:
http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/infmts.htm#trn_fmt_name_0
"""

import re

TRANSCRIPT_PATTERN = re.compile(r'(.*) \((.*?)\)')


def write(path, entries):
    """
    Writes an transcription file.

    Args:
        path (str): Path to write the file to.
        entries (dict): List with entries to write.

    Example::

        >>> data = {
        >>>     'utt-1': 'sie',
        >>>     'utt-2': 'hallo',
        >>> }
        >>>
        >>> write_label_file('/some/path/to/file.txt', data)
    """
    sorted_entries = sorted(entries.items(), key=lambda x: x[0])
    lines = ['{} ({})'.format(x[1], x[0]) for x in sorted_entries]

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def read(path):
    """
    Read the labels from a transcription file.

    Args:
        path (str): Path to the label file.

    Returns:
        dict: Dictionary of transcriptions (utt-idx: transcription)

    Example::

        >>> read_label_file('/path/to/label/file.txt')
        {
            'utt-1': 'sie',
            'utt-2': 'hallo'
        }
    """

    entries = {}

    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip()

            m = TRANSCRIPT_PATTERN.match(line)

            if m is not None:
                utt_idx = m.group(2).strip()
                transcription = m.group(1).strip()
                entries[utt_idx] = transcription
            else:
                raise ValueError('Failed to parse line of trn file: {}'.format(line))

    return entries
