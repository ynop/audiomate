from pingu.corpus.subset import utils


class TestUtils:

    def test_absolute_proportions(self):
        res = utils.absolute_proportions({
            'a': 0.6,
            'b': 0.2,
            'c': 0.2
        }, 120)

        assert res['a'] == 72
        assert res['b'] == 24
        assert res['c'] == 24

    def test_get_identifiers_randomly_splitted(self):
        res = utils.split_identifiers(identifiers=[
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'v', 't'
        ], proportions={
            'a': 0.3333,
            'b': 0.6666
        })

        assert len(res['a']) == 4
        assert len(res['b']) == 8
        assert len(set(res['a'] + res['b'])) == 12

    def test_get_identifiers_splitted_by_weights_single_category(self):
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

    def test_get_identifiers_splitted_by_weights(self):
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
