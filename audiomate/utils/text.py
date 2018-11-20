"""
This module contains any functions for working with text/strings/punctuation.
"""

import re


def remove_punctuation(text, exceptions=[]):
    """
    Return a string with punctuation removed.

    Parameters:
        text (str): The text to remove punctuation from.
        exceptions (list): List of symbols to keep in the given text.

    Return:
        str: The input text without the punctuation.
    """

    all_but = [
        r'\w',
        r'\s'
    ]

    all_but.extend(exceptions)

    pattern = '[^{}]'.format(''.join(all_but))

    return re.sub(pattern, '', text)


def starts_with_prefix_in_list(text, prefixes):
    """
    Return True if the given string starts with one of the prefixes in the given list, otherwise
    return False.

    Arguments:
        text (str): Text to check for prefixes.
        prefixes (list): List of prefixes to check for.

    Returns:
        bool: True if the given text starts with any of the given prefixes, otherwise False.
    """
    for prefix in prefixes:
        if text.startswith(prefix):
            return True
    return False
