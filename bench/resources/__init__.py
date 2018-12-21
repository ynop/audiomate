import random

import audiomate
from audiomate import issuers
from audiomate import tracks
from audiomate import annotations


def generate_corpus(n_issuers,
                    n_tracks_per_issuer,
                    n_utts_per_track,
                    n_ll_per_utt,
                    n_label_per_ll,
                    rand=None):
    """
    Generate a corpus with mock data.
    """
    corpus = audiomate.Corpus()

    for issuer in generate_issuers(n_issuers, rand):
        corpus.import_issuers(issuer)

        n_tracks = rand.randint(*n_tracks_per_issuer)
        tracks = generate_tracks(n_tracks, rand)
        corpus.import_tracks(tracks)

        n_utts = rand.randint(*n_utts_per_track)
        for track in tracks:
            utts = generate_utterances(
                track,
                issuer,
                n_utts,
                n_ll_per_utt,
                n_label_per_ll,
                rand
            )

            corpus.import_utterances(utts)

    return corpus


def generate_issuers(n, rand=None):
    if rand is None:
        rand = random.Random()

    items = []

    for issuer_index in range(n):
        issuer_idx = 'issuer-{}'.format(issuer_index)

        issuer_type = rand.randint(1, 3)

        if issuer_type == 1:
            issuer = issuers.Speaker(
                issuer_idx,
                gender=issuers.Gender.UNKNOWN,
                age_group=issuers.AgeGroup.CHILD,
                native_language='de'
            )
        elif issuer_type == 2:
            issuer = issuers.Artist(issuer_idx, 'badam')
        else:
            issuer = issuers.Issuer(issuer_idx)

        items.append(issuer)

    return items


def generate_tracks(n, rand=None):
    if rand is None:
        rand = random.Random()

    items = []

    for i in range(n):
        track_idx = 'track-{}'.format(i)
        path = '/fake/{}.wav'.format(track_idx)
        track = tracks.FileTrack(track_idx, path)

        items.append(track)

    return items


def generate_utterances(track, issuer, n, n_ll_range, n_label_range, rand=None):
    if rand is None:
        rand = random.Random()

    items = []

    for i in range(n):
        utt_idx = '{}-utt-{}'.format(track.idx, i)
        start = rand.random() * 3
        end = 3 + rand.random() * 8

        utt = tracks.Utterance(
            utt_idx,
            track,
            issuer=issuer,
            start=start,
            end=end
        )
        n_ll = rand.randint(*n_ll_range)

        for ll in generate_label_lists(n_ll, n_label_range, rand=rand):
            utt.set_label_list(ll)

        items.append(utt)

    return items


def generate_label_lists(n, n_label_range, rand=None):
    if rand is None:
        rand = random.Random()

    items = []

    for i in range(n):
        n_labels = rand.randint(*n_label_range)
        ll_idx = 'll-{}'.format(i)
        ll = annotations.LabelList(
            idx=ll_idx,
            labels=generate_labels(n_labels)
        )

        items.append(ll)

    return items


def generate_labels(n):
    items = []

    for i in range(n):
        label = annotations.Label('label-{}'.format(i))
        items.append(label)

    return items
