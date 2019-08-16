import random

from audiomate.corpus.subset import utils


def run_split_identifiers():
    identifiers = list(range(10000))
    proportions = {'a': 0.2, 'b': 0.2, 'c': 0.6}
    utils.split_identifiers(identifiers, proportions)


def test_split_identifiers(benchmark):
    benchmark(run_split_identifiers)


def run_get_identifiers_splitted_by_weights():
    identifiers = {}

    for i in range(100000):
        identifiers[str(i)] = {
            'a': random.randint(2, 10),
            'b': random.randint(2, 10),
            'c': random.randint(2, 10)
        }

    proportions = {'a': 0.2, 'b': 0.2, 'c': 0.6}

    utils.get_identifiers_splitted_by_weights(identifiers, proportions)


def test_get_identifiers_splitted_by_weights(benchmark):
    benchmark(run_get_identifiers_splitted_by_weights)
