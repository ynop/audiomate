import statistics

from audiomate.corpus.subset import utils


def test_absolute_proportions():
    res = utils.absolute_proportions({
        'a': 0.6,
        'b': 0.2,
        'c': 0.2
    }, 120)

    assert res['a'] == 72
    assert res['b'] == 24
    assert res['c'] == 24


def test_get_identifiers_randomly_splitted():
    res = utils.split_identifiers(identifiers=[
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'v', 't'
    ], proportions={
        'a': 0.3333,
        'b': 0.6666
    })

    assert len(res['a']) == 4
    assert len(res['b']) == 8
    assert len(set(res['a'] + res['b'])) == 12


def test_get_identifiers_splitted_by_weights_single_category():
    identifiers = {
        'a': {'mi': 3},
        'b': {'mi': 4},
        'c': {'mi': 6},
        'd': {'mi': 1},
        'e': {'mi': 4},
        'f': {'mi': 5},
        'g': {'mi': 3}
    }

    proportions = {
        'train': 0.5,
        'test': 0.25,
        'dev': 0.25
    }

    res = utils.get_identifiers_splitted_by_weights(identifiers=identifiers,
                                                    proportions=proportions)

    assert len(res['train']) > 0
    assert len(res['test']) > 0
    assert len(res['dev']) > 0
    assert len(identifiers) == sum([len(x) for x in res.values()])


def test_get_identifiers_splitted_by_weights():
    identifiers = {
        'a': {'mi': 3, 'ma': 2, 'mu': 1},
        'b': {'mi': 4, 'ma': 5, 'mu': 4},
        'c': {'mi': 6, 'ma': 4, 'mu': 3},
        'd': {'mi': 1, 'ma': 3, 'mu': 2},
        'e': {'mi': 4, 'ma': 1, 'mu': 5},
        'f': {'mi': 5, 'ma': 4, 'mu': 3},
        'g': {'mi': 3, 'ma': 4, 'mu': 5}
    }

    proportions = {
        'train': 0.5,
        'test': 0.25,
        'dev': 0.25
    }

    res = utils.get_identifiers_splitted_by_weights(identifiers=identifiers,
                                                    proportions=proportions)

    assert len(res['train']) > 0
    assert len(res['test']) > 0
    assert len(res['dev']) > 0
    assert len(identifiers) == sum([len(x) for x in res.values()])


def test_select_balanced_subset():
    categories = ['a', 'b', 'c']
    items = {
        'utt-1': {'a': 1, 'b': 0, 'c': 0},
        'utt-2': {'a': 1, 'c': 2},
        'utt-3': {'a': 1, 'b': 1, 'c': 1},
        'utt-4': {'a': 1, 'b': 1, 'c': 0},
        'utt-5': {'b': 1, 'c': 0},
        'utt-6': {'b': 2, 'c': 1},
        'utt-7': {'a': 1, 'b': 0, 'c': 1},
        'utt-8': {'a': 1, 'b': 1, 'c': 0},
        'utt-9': {'a': 1, 'b': 0, 'c': 0},
        'utt-10': {'c': 2},
        'utt-11': {'a': 1, 'b': 1, 'c': 1},
        'utt-12': {'b': 1, 'c': 0},
        'utt-13': {'b': 1, 'c': 0},
        'utt-14': {'b': 2, 'c': 1},
        'utt-15': {'a': 1, 'b': 0, 'c': 1},
        'utt-16': {'a': 1, 'b': 1, 'c': 0}
    }

    item_ids = utils.select_balanced_subset(items, 10, categories, seed=33)
    weights = [0 for __ in categories]

    for item_id in item_ids:
        for cat, weight in items[item_id].items():
            weights[categories.index(cat)] += weight

    assert len(item_ids) == 10
    assert statistics.variance(weights) <= 1
