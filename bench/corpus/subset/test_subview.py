import random

from audiomate.corpus import subset

from bench import resources


def run(corpus, filters):
    subview = subset.Subview(corpus, filters)
    subview.utterances
    subview.issuers
    subview.tracks
    subview.num_utterances
    subview.num_issuers
    subview.num_tracks


def test_subview(benchmark):
    corpus = resources.generate_corpus(
        200,
        (5, 10),
        (1, 5),
        (0, 6),
        (1, 20),
        random.Random(x=234)
    )

    random.seed(200)
    filtered_utts = random.choices(list(corpus.utterances.keys()), k=20000)
    filters = [
        subset.MatchingUtteranceIdxFilter(filtered_utts)
    ]

    benchmark(run, corpus, filters)
