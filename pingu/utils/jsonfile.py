"""
This module contains functions for reading and writing json files.
"""

import json


def write_json_to_file(path, data):
    """
    Writes data as json to file.

    Parameters:
        path (str): Path to write to
        data (dict, list): Data
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def read_json_file(path):
    """
    Reads and return the data from the json file at the given path.

    Parameters:
        path (str): Path to read

    Returns:
        dict,list: The read json as dict/list.
    """

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data
