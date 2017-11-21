import re


def remove_punctuation(text, exceptions=[]):
    """
    Return a string with punctuation removed.

    Parameters:
        text: The text to remove punctuation from.
        exceptions: List of symbols to keep in the given text.
    """

    all_but = [
        '\w',
        '\s'
    ]

    all_but.extend(exceptions)

    pattern = '[^{}]'.format(''.join(all_but))

    return re.sub(pattern, '', text)


def starts_with_prefix_in_list(text, prefixes):
    """
    Return True if the given string starts with one of the prefixes in the given list, otherwise return False.

    Arguments:
        text: Text to check for prefixes.
        prefixes: List of prefixes to check for.
    """
    for prefix in prefixes:
        if text.startswith(prefix):
            return True
    return False
