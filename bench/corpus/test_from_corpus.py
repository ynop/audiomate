import random

import audiomate

from bench import resources


def run(source_corpus):
    audiomate.Corpus.from_corpus(source_corpus)


def test_from_corpus(benchmark):
    source_corpus = resources.generate_corpus(
        200,
        (5, 5),
        (5, 5),
        (4, 4),
        (4, 4),
        random.Random(x=234)
    )

    benchmark(run, source_corpus)


if __name__ == '__main__':
    source_corpus = resources.generate_corpus(
        200,
        (5, 10),
        (1, 5),
        (0, 6),
        (1, 20),
        random.Random(x=234)
    )
    audiomate.Corpus.from_corpus(source_corpus)
