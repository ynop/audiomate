import random

from audiomate.corpus import subset

from bench import resources


def run(splitter):
    out = splitter.split_by_length_of_utterances(
        {
            'train': 0.7,
            'test': 0.15,
            'dev': 0.15
        },
        separate_issuers=True
    )

    assert len(out) == 3


def test_split_by_length_of_utterances(benchmark):
        corpus = resources.generate_corpus(
            179,
            (250, 500),
            (1, 9),
            (0, 6),
            (1, 20),
            random.Random(x=234)
        )

        splitter = subset.Splitter(corpus, random_seed=324)

        benchmark(run, splitter)
