import random

import audiomate

from bench import resources


def run(target_corpus, merge_corpus):
    target_corpus.merge_corpus(merge_corpus)


def test_merge_corpus(benchmark):
    target_corpus = audiomate.Corpus()
    merge_corpus = resources.generate_corpus(
        200,
        (5, 10),
        (1, 5),
        (0, 6),
        (1, 20),
        random.Random(x=234)
    )

    benchmark(run, target_corpus, merge_corpus)
