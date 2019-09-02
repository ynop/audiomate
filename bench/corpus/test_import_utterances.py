import random

import audiomate

from bench import resources


def run(target_corpus, utterances):
    target_corpus.import_utterances(utterances)


def prepare():
    rand = random.Random(x=234)

    target_corpus = audiomate.Corpus()

    issuers = resources.generate_issuers(1000, rand=rand)
    target_corpus.import_issuers(issuers)

    tracks = resources.generate_tracks(1000, rand=rand)
    target_corpus.import_tracks(tracks)

    utterances = []

    for issuer, track in zip(issuers, tracks):
        utts = resources.generate_utterances(track, issuer, 10, (3, 3), (3, 3), rand=rand)
        utterances.extend(utts)

    return target_corpus, utterances


def test_import_utterances(benchmark):
    target_corpus, utterances = prepare()
    benchmark(run, target_corpus, utterances)


if __name__ == '__main__':
    target_corpus, utterances = prepare()
    run(target_corpus, utterances)
