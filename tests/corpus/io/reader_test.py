import os
from collections import namedtuple

from audiomate import tracks
from audiomate import issuers
from audiomate import annotations

import pytest


ExpFileTrack = namedtuple(
    'FileTrack',
    ['idx', 'path']
)
ExpContainerTrack = namedtuple(
    'ContainerTrack',
    ['idx', 'key', 'container_path']
)

ExpIssuer = namedtuple(
    'Issuer',
    ['idx', 'num_utterances']
)

ExpSpeaker = namedtuple(
    'Speaker',
    ['idx', 'num_utterances', 'gender', 'age_group', 'native_lang']
)

ExpArtist = namedtuple(
    'Artist',
    ['idx', 'num_utterances', 'name']
)

ExpUtterance = namedtuple(
    'Utterance',
    ['idx', 'track_idx', 'issuer_idx', 'start', 'end']
)

ExpLabelList = namedtuple(
    'LabelList',
    ['ll_idx', 'num_labels']
)

ExpLabel = namedtuple(
    'Label',
    ['ll_idx', 'value', 'start', 'end']
)

ExpSubview = namedtuple(
    'Subview',
    ['idx', 'utterance_ids']
)


class CorpusReaderTest:
    """
    Base class for testing a corpus reader.

    All the basic components can be tested by defining the expected assets
    using the EXPECTED-variables.
    """

    FILE_TRACK_BASE_PATH = None

    EXPECTED_NUMBER_OF_TRACKS = None
    EXPECTED_TRACKS = []

    EXPECTED_NUMBER_OF_UTTERANCES = None
    EXPECTED_UTTERANCES = []

    EXPECTED_NUMBER_OF_ISSUERS = None
    EXPECTED_ISSUERS = []

    # Expected Label-Lists/Labels per utterance
    EXPECTED_LABEL_LISTS = {}
    EXPECTED_LABELS = {}

    EXPECTED_NUMBER_OF_SUBVIEWS = None
    EXPECTED_SUBVIEWS = []

    def load(self):
        pass

    # ----------------------------------------------------------------------
    # TRACKS
    # ----------------------------------------------------------------------

    def test_number_of_tracks(self):
        assert self.EXPECTED_NUMBER_OF_TRACKS is not None, \
            'Expected number of tracks has to be defined'

        corpus = self.load()

        assert corpus.num_tracks == self.EXPECTED_NUMBER_OF_TRACKS, \
            'Number of tracks in corpus is wrong'

    def test_tracks(self):
        if len(self.EXPECTED_TRACKS) == 0:
            pytest.skip('No tracks expected')

        corpus = self.load()

        for exp in self.EXPECTED_TRACKS:
            idx = exp.idx

            assert idx in corpus.tracks.keys(), \
                'track with idx "{}" is missing'.format(idx)

            actual = corpus.tracks[idx]

            assert actual.idx == idx, \
                'track idx "{}" differs from track-dictionary idx "{}"'.format(
                    idx, actual.idx)

            if isinstance(exp, ExpFileTrack):
                assert isinstance(actual, tracks.FileTrack), \
                    'Track "{}" has a wrong type'.format(idx)

                exp_path = exp.path

                if self.FILE_TRACK_BASE_PATH is not None:
                    exp_path = os.path.join(
                        self.FILE_TRACK_BASE_PATH,
                        exp.path
                    )
                assert exp_path == actual.path, \
                    'Path of file "{}" is wrong'.format(idx)
            elif isinstance(exp, ExpFileTrack):
                assert isinstance(actual, tracks.ContainerTrack), \
                    'Track "{}" has a wrong type'.format(idx)
                assert actual.key == exp.key, \
                    'Key of container-track "{}" is wrong'.format(idx)
                assert actual.container.path == exp.key, \
                    'Container-path of track "{}" is wrong'.format(idx)

    # ----------------------------------------------------------------------
    # UTTERANCES
    # ----------------------------------------------------------------------

    def test_number_of_utterances(self):
        assert self.EXPECTED_NUMBER_OF_UTTERANCES is not None, \
            'Expected number of utterances has to be defined'

        corpus = self.load()

        assert corpus.num_utterances == self.EXPECTED_NUMBER_OF_UTTERANCES, \
            'Number of utterances in corpus is wrong'

    def test_utterances(self):
        if len(self.EXPECTED_UTTERANCES) == 0:
            pytest.skip('No utterances expected')

        corpus = self.load()

        for exp in self.EXPECTED_UTTERANCES:
            idx = exp.idx

            assert idx in corpus.utterances.keys(), \
                'Utterance with idx "{}" is missing'.format(idx)
            assert corpus.utterances[idx].idx == idx, \
                ('utterance idx "{}" differs from'
                 'utterance-dictionary idx "{}"').format(
                idx, corpus.tracks[idx].idx
            )

            actual = corpus.utterances[idx]
            assert actual.start == exp.start, \
                'Start of utterance is wrong'
            assert actual.end == exp.end, \
                'End of utterance is wrong'

            if exp.issuer_idx is None:
                assert actual.issuer is None, \
                    'Utterance is expected to have no issuer'
            else:
                assert actual.issuer.idx == exp.issuer_idx, \
                    'Issuer-idx of utterance {} is wrong'.format(idx)

    # ----------------------------------------------------------------------
    # ISSUERS
    # ----------------------------------------------------------------------

    def test_number_of_issuers(self):
        assert self.EXPECTED_NUMBER_OF_ISSUERS is not None, \
            'Expected number of issuers has to be defined'

        corpus = self.load()

        assert corpus.num_issuers == self.EXPECTED_NUMBER_OF_ISSUERS, \
            'Number of issuers in corpus is wrong'

    def test_issuers(self):
        corpus = self.load()

        for exp in self.EXPECTED_ISSUERS:
            idx = exp.idx

            assert idx in corpus.issuers.keys(), \
                'Issuer with idx "{}" is missing'.format(idx)

            actual = corpus.issuers[idx]

            assert actual.idx == idx, \
                ('Issuer idx "{}" differs from'
                 'issuer-dictionary idx "{}"').format(
                    idx, actual.idx
                )
            assert len(actual.utterances) == exp.num_utterances, \
                'Issuer "{}" has the wrong num. of utterances'.format(idx)

            if isinstance(exp, ExpSpeaker):
                assert isinstance(actual, issuers.Speaker), \
                    'Issuer "{}" has a wrong type'.format(idx)
                assert actual.gender == exp.gender, \
                    'Gender of speaker "{}" is wrong'.format(idx)
                assert actual.age_group == exp.age_group, \
                    'AgeGroup of speaker "{}" is wrong'.format(idx)
                assert actual.native_language == exp.native_lang, \
                    'Native language of speaker "{}" is wrong'.format(idx)
            elif isinstance(exp, ExpArtist):
                assert isinstance(actual, issuers.Artist), \
                    'Issuer "{}" has a wrong type'.format(idx)
                assert actual.name == exp.name, \
                    'Name of artist "{}" is wrong'.format(idx)
            elif isinstance(exp, ExpIssuer):
                assert isinstance(actual, issuers.Issuer), \
                    'Issuer "{}" has a wrong type'.format(idx)

    # ----------------------------------------------------------------------
    # LABEL / LABELLISTS
    # ----------------------------------------------------------------------

    def test_label_lists(self):
        corpus = self.load()

        for utt_idx, utt_label_lists in self.EXPECTED_LABEL_LISTS.items():
            utt = corpus.utterances[utt_idx]

            for exp_ll in utt_label_lists:
                idx = exp_ll.ll_idx

                assert idx in utt.label_lists.keys(), \
                    'LabelList with idx "{}" is missing'.format(idx)

                actual_ll = utt.label_lists[idx]

                assert actual_ll.idx == idx, \
                    ('LabelList idx "{}" differs from'
                     'labellists-dictionary idx "{}"').format(
                        idx, actual_ll.idx
                    )
                assert len(actual_ll) == exp_ll.num_labels, \
                    '[{}] Number of labels in label-list "{}" is wrong'.format(utt_idx, idx)

    def test_labels(self):
        corpus = self.load()

        for utt_idx, utt_labels in self.EXPECTED_LABELS.items():
            utt = corpus.utterances[utt_idx]

            for exp in utt_labels:
                ll = utt.label_lists[exp.ll_idx]
                expected = annotations.Label(exp.value, exp.start, exp.end)

                assert expected in list(ll), \
                    '{} not in LabelList {}'.format(expected, ll.idx)

    # ----------------------------------------------------------------------
    # SUBVIEWS
    # ----------------------------------------------------------------------

    def test_number_of_subviews(self):
        assert self.EXPECTED_NUMBER_OF_SUBVIEWS is not None, \
            'Expected number of subviews has to be defined'

        corpus = self.load()

        assert corpus.num_subviews == self.EXPECTED_NUMBER_OF_SUBVIEWS, \
            'Number of subviews in corpus is wrong'

    def test_subviews(self):
        corpus = self.load()

        for exp in self.EXPECTED_SUBVIEWS:
            idx = exp.idx

            assert idx in corpus.subviews.keys(), \
                'Subview with idx "{}" is missing'.format(idx)

            actual = corpus.subviews[idx]

            assert list(sorted(actual.utterances.keys())) == \
                list(sorted(exp.utterance_ids)), \
                'Subviews has mismatches of utterance ids'
