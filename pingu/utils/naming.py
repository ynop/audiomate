import random
import string


def index_name_if_in_list(name, name_list, suffix='', prefix=''):
    """
    Find a unique name by adding an index to the name so it is unique within the given list.

    Parameters:
        name: Name
        name_list: List of names that the new name must differ from.
        suffix: The suffix to append after the index.
        prefix: The prefix to append in front of the index.
    """
    new_name = '{}'.format(name)
    index = 1

    while new_name in name_list:
        new_name = '{}_{}{}{}'.format(name, prefix, index, suffix)
        index += 1

    return new_name


def generate_name(length=15, not_in=None):
    """
    Generates a random string of lowercase letters with the given length.

    Parameters:
        length: Length of the string to output.
        not_in: Only return a string not in the given iterator.
    """
    value = ''.join(random.choice(string.ascii_lowercase) for i in range(length))

    while (not_in is not None) and (value in not_in):
        value = ''.join(random.choice(string.ascii_lowercase) for i in range(length))

    return value
