import random

import pytest
from unittest import mock

from audiomate import annotations
from audiomate.corpus.subset import splitting
from tests import resources


INITIAL_SEED = 99


@pytest.fixture
def corpus():
    return resources.create_multi_label_corpus()


@pytest.fixture
def splitter():
    corpus = resources.create_multi_label_corpus()
    return splitting.Splitter(corpus, random_seed=INITIAL_SEED)


class TestSplitter:

    #
    # SPLIT()
    #

    @mock.patch('audiomate.corpus.subset.utils.split_identifiers')
    def test_split(self, split_mock, splitter):
        split_mock.return_value = {
            'train': ['utt-1', 'utt-3'],
            'test': ['utt-3', 'utt-4']
        }
        res = splitter.split({'train': 0.5, 'test': 0.5})

        assert res['train'].utterances.keys() == {'utt-1', 'utt-3'}
        assert res['test'].utterances.keys() == {'utt-3', 'utt-4'}

        split_mock.assert_called_with(
            list(splitter.corpus.utterances.keys()),
            {'train': 0.5, 'test': 0.5},
            seed=mock.ANY
        )

    @mock.patch('audiomate.corpus.subset.utils.split_identifiers')
    def test_split_passes_seed(self, split_mock, splitter):
        splitter.split(
            {'train': 0.5, 'test': 0.5},
            separate_issuers=False
        )

        r = random.Random(INITIAL_SEED)

        split_mock.assert_called_with(
            mock.ANY,
            mock.ANY,
            seed=r.random()
        )

    @mock.patch('audiomate.corpus.subset.utils.get_identifiers_splitted_by_weights')
    def test_split_issuer_separated(self, split_mock, splitter):
        split_mock.return_value = {
            'train': ['spk-1', 'spk-2'],
            'test': ['spk-3']
        }
        res = splitter.split(
            {'train': 0.5, 'test': 0.5},
            separate_issuers=True
        )

        assert res['train'].utterances.keys() == {
            'utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'
        }
        assert res['test'].utterances.keys() == {
            'utt-6', 'utt-7', 'utt-8'
        }

        split_mock.assert_called_with(
            {
                'spk-1': {'count': 2},
                'spk-2': {'count': 3},
                'spk-3': {'count': 3},
            },
            {'train': 0.5, 'test': 0.5},
            seed=mock.ANY
        )

    @mock.patch('audiomate.corpus.subset.utils.get_identifiers_splitted_by_weights')
    def test_split_passes_issuer_separated_seed(self, split_mock, splitter):
        splitter.split(
            {'train': 0.5, 'test': 0.5},
            separate_issuers=True
        )

        r = random.Random(INITIAL_SEED)

        split_mock.assert_called_with(
            mock.ANY,
            mock.ANY,
            seed=r.random()
        )

    #
    # SPLIT_BY_AUDIO_DURATION()
    #

    @mock.patch('audiomate.corpus.subset.utils.get_identifiers_splitted_by_weights')
    def test_split_by_audio_duration(self, split_mock, splitter):
        split_mock.return_value = {
            'train': ['utt-1', 'utt-3'],
            'test': ['utt-3', 'utt-4'],
            'dev': ['utt-5', 'utt-6'],
        }
        res = splitter.split_by_audio_duration({'train': 0.6, 'test': 0.2, 'dev': 0.2})

        assert res['train'].utterances.keys() == {'utt-1', 'utt-3'}
        assert res['test'].utterances.keys() == {'utt-3', 'utt-4'}
        assert res['dev'].utterances.keys() == {'utt-5', 'utt-6'}

        split_mock.assert_called_with(
            {
                'utt-1': {'duration': 2595},
                'utt-2': {'duration': 2595},
                'utt-3': {'duration': 15000},
                'utt-4': {'duration': 10000},
                'utt-5': {'duration': 15000},
                'utt-6': {'duration': 15000},
                'utt-7': {'duration': 10000},
                'utt-8': {'duration': 15000},
            },
            {'train': 0.6, 'test': 0.2, 'dev': 0.2},
            seed=mock.ANY
        )

    @mock.patch('audiomate.corpus.subset.utils.get_identifiers_splitted_by_weights')
    def test_split_by_audio_duration_passes_seed(self, split_mock, splitter):
        splitter.split_by_audio_duration(
            {'train': 0.5, 'test': 0.5},
            separate_issuers=False
        )

        r = random.Random(INITIAL_SEED)

        split_mock.assert_called_with(
            mock.ANY,
            mock.ANY,
            seed=r.random()
        )

    @mock.patch('audiomate.corpus.subset.utils.get_identifiers_splitted_by_weights')
    def test_split_by_audio_duration_issuer_separated(self, split_mock, splitter):
        split_mock.return_value = {
            'train': ['spk-1', 'spk-2'],
            'test': ['spk-3']
        }
        res = splitter.split_by_audio_duration(
            {'train': 0.5, 'test': 0.5},
            separate_issuers=True
        )

        assert res['train'].utterances.keys() == {
            'utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'
        }
        assert res['test'].utterances.keys() == {
            'utt-6', 'utt-7', 'utt-8'
        }

        split_mock.assert_called_with(
            {
                'spk-1': {'duration': 5190},
                'spk-2': {'duration': 40000},
                'spk-3': {'duration': 40000},
            },
            {'train': 0.5, 'test': 0.5},
            seed=mock.ANY
        )

    @mock.patch('audiomate.corpus.subset.utils.get_identifiers_splitted_by_weights')
    def test_split_by_audio_duration_passes_issuer_separated_seed(self, split_mock, splitter):
        splitter.split_by_audio_duration(
            {'train': 0.5, 'test': 0.5},
            separate_issuers=True
        )

        r = random.Random(INITIAL_SEED)

        split_mock.assert_called_with(
            mock.ANY,
            mock.ANY,
            seed=r.random()
        )

    #
    # SPLIT_BY_LABEL_LENGTH()
    #

    @mock.patch('audiomate.corpus.subset.utils.get_identifiers_splitted_by_weights')
    def test_split_by_label_length(self, split_mock, splitter):
        split_mock.return_value = {
            'train': ['utt-1', 'utt-3'],
            'test': ['utt-3', 'utt-4'],
            'dev': ['utt-5', 'utt-6'],
        }
        res = splitter.split_by_label_length({'train': 0.6, 'test': 0.2, 'dev': 0.2})

        assert res['train'].utterances.keys() == {'utt-1', 'utt-3'}
        assert res['test'].utterances.keys() == {'utt-3', 'utt-4'}
        assert res['dev'].utterances.keys() == {'utt-5', 'utt-6'}

        split_mock.assert_called_with(
            {
                'utt-1': {'length': 16},
                'utt-2': {'length': 16},
                'utt-3': {'length': 11},
                'utt-4': {'length': 16},
                'utt-5': {'length': 6},
                'utt-6': {'length': 16},
                'utt-7': {'length': 11},
                'utt-8': {'length': 5},
            },
            {'train': 0.6, 'test': 0.2, 'dev': 0.2},
            seed=mock.ANY
        )

    @mock.patch('audiomate.corpus.subset.utils.get_identifiers_splitted_by_weights')
    def test_split_by_label_length_passes_seed(self, split_mock, splitter):
        splitter.split_by_label_length(
            {'train': 0.5, 'test': 0.5},
            separate_issuers=False
        )

        r = random.Random(INITIAL_SEED)

        split_mock.assert_called_with(
            mock.ANY,
            mock.ANY,
            seed=r.random()
        )

    @mock.patch('audiomate.corpus.subset.utils.get_identifiers_splitted_by_weights')
    def test_split_by_label_length_only_uses_given_label_list(self, split_mock):
        corpus = resources.create_multi_label_corpus()
        splitter = splitting.Splitter(corpus, random_seed=INITIAL_SEED)

        for utt in corpus.utterances.values():
            utt.set_label_list(annotations.LabelList.create_single(
                'another label', idx='some-idx'
            ))

        split_mock.return_value = {
            'train': ['utt-1', 'utt-3'],
            'test': ['utt-3', 'utt-4'],
            'dev': ['utt-5', 'utt-6'],
        }
        res = splitter.split_by_label_length(
            {'train': 0.6, 'test': 0.2, 'dev': 0.2},
            label_list_idx='default'
        )

        assert res['train'].utterances.keys() == {'utt-1', 'utt-3'}
        assert res['test'].utterances.keys() == {'utt-3', 'utt-4'}
        assert res['dev'].utterances.keys() == {'utt-5', 'utt-6'}

        split_mock.assert_called_with(
            {
                'utt-1': {'length': 16},
                'utt-2': {'length': 16},
                'utt-3': {'length': 11},
                'utt-4': {'length': 16},
                'utt-5': {'length': 6},
                'utt-6': {'length': 16},
                'utt-7': {'length': 11},
                'utt-8': {'length': 5},
            },
            {'train': 0.6, 'test': 0.2, 'dev': 0.2},
            seed=mock.ANY
        )

    @mock.patch('audiomate.corpus.subset.utils.get_identifiers_splitted_by_weights')
    def test_split_by_label_length_issuer_separated(self, split_mock, splitter):
        split_mock.return_value = {
            'train': ['spk-1', 'spk-2'],
            'test': ['spk-3']
        }
        res = splitter.split_by_label_length(
            {'train': 0.5, 'test': 0.5},
            separate_issuers=True
        )

        assert res['train'].utterances.keys() == {
            'utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'
        }
        assert res['test'].utterances.keys() == {
            'utt-6', 'utt-7', 'utt-8'
        }

        split_mock.assert_called_with(
            {
                'spk-1': {'length': 32},
                'spk-2': {'length': 33},
                'spk-3': {'length': 32},
            },
            {'train': 0.5, 'test': 0.5},
            seed=mock.ANY
        )

    @mock.patch('audiomate.corpus.subset.utils.get_identifiers_splitted_by_weights')
    def test_split_by_label_length_passes_issuer_separated_seed(self, split_mock, splitter):
        splitter.split_by_label_length(
            {'train': 0.5, 'test': 0.5},
            separate_issuers=True
        )

        r = random.Random(INITIAL_SEED)

        split_mock.assert_called_with(
            mock.ANY,
            mock.ANY,
            seed=r.random()
        )

    @mock.patch('audiomate.corpus.subset.utils.get_identifiers_splitted_by_weights')
    def test_split_by_label_length_issuer_separated_only_uses_given_label_list(self, split_mock):
        corpus = resources.create_multi_label_corpus()
        splitter = splitting.Splitter(corpus, random_seed=INITIAL_SEED)

        for utt in corpus.utterances.values():
            utt.set_label_list(annotations.LabelList.create_single(
                'another label', idx='some-idx'
            ))

        split_mock.return_value = {
            'train': ['spk-1', 'spk-2'],
            'test': ['spk-3']
        }
        res = splitter.split_by_label_length(
            {'train': 0.5, 'test': 0.5},
            label_list_idx='default',
            separate_issuers=True
        )

        assert res['train'].utterances.keys() == {
            'utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'
        }
        assert res['test'].utterances.keys() == {
            'utt-6', 'utt-7', 'utt-8'
        }

        split_mock.assert_called_with(
            {
                'spk-1': {'length': 32},
                'spk-2': {'length': 33},
                'spk-3': {'length': 32},
            },
            {'train': 0.5, 'test': 0.5},
            seed=mock.ANY
        )

    #
    # SPLIT_BY_LABEL_OCCURENCE()
    #

    @mock.patch('audiomate.corpus.subset.utils.get_identifiers_splitted_by_weights')
    def test_split_by_label_occurence(self, split_mock, splitter):
        split_mock.return_value = {
            'train': ['utt-1', 'utt-3'],
            'test': ['utt-3', 'utt-4'],
            'dev': ['utt-5', 'utt-6'],
        }
        res = splitter.split_by_label_occurence({'train': 0.6, 'test': 0.2, 'dev': 0.2})

        assert res['train'].utterances.keys() == {'utt-1', 'utt-3'}
        assert res['test'].utterances.keys() == {'utt-3', 'utt-4'}
        assert res['dev'].utterances.keys() == {'utt-5', 'utt-6'}

        split_mock.assert_called_with(
            {
                'utt-1': {'music': 2, 'speech': 1},
                'utt-2': {'music': 2, 'speech': 1},
                'utt-3': {'music': 1, 'speech': 1},
                'utt-4': {'music': 2, 'speech': 1},
                'utt-5': {'speech': 1},
                'utt-6': {'music': 2, 'speech': 1},
                'utt-7': {'music': 1, 'speech': 1},
                'utt-8': {'music': 1},
            },
            {'train': 0.6, 'test': 0.2, 'dev': 0.2},
            seed=mock.ANY
        )

    @mock.patch('audiomate.corpus.subset.utils.get_identifiers_splitted_by_weights')
    def test_split_by_label_occurence_passes_seed(self, split_mock, splitter):
        splitter.split_by_label_occurence(
            {'train': 0.5, 'test': 0.5},
            separate_issuers=False
        )

        r = random.Random(INITIAL_SEED)

        split_mock.assert_called_with(
            mock.ANY,
            mock.ANY,
            seed=r.random()
        )

    @mock.patch('audiomate.corpus.subset.utils.get_identifiers_splitted_by_weights')
    def test_split_by_label_occurence_issuer_separated(self, split_mock, splitter):
        split_mock.return_value = {
            'train': ['spk-1', 'spk-2'],
            'test': ['spk-3']
        }
        res = splitter.split_by_label_occurence(
            {'train': 0.5, 'test': 0.5},
            separate_issuers=True
        )

        assert res['train'].utterances.keys() == {
            'utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'
        }
        assert res['test'].utterances.keys() == {
            'utt-6', 'utt-7', 'utt-8'
        }

        split_mock.assert_called_with(
            {
                'spk-1': {'music': 4, 'speech': 2},
                'spk-2': {'music': 3, 'speech': 3},
                'spk-3': {'music': 4, 'speech': 2},
            },
            {'train': 0.5, 'test': 0.5},
            seed=mock.ANY
        )

    @mock.patch('audiomate.corpus.subset.utils.get_identifiers_splitted_by_weights')
    def test_split_by_label_occurence_passes_issuer_separated_seed(self, split_mock, splitter):
        splitter.split_by_label_occurence(
            {'train': 0.5, 'test': 0.5},
            separate_issuers=True
        )

        r = random.Random(INITIAL_SEED)

        split_mock.assert_called_with(
            mock.ANY,
            mock.ANY,
            seed=r.random()
        )

    #
    # SPLIT_BY_LABEL_DURATION()
    #

    @mock.patch('audiomate.corpus.subset.utils.get_identifiers_splitted_by_weights')
    def test_split_by_label_duration(self, split_mock, splitter):
        split_mock.return_value = {
            'train': ['utt-1', 'utt-3'],
            'test': ['utt-3', 'utt-4'],
            'dev': ['utt-5', 'utt-6'],
        }
        res = splitter.split_by_label_duration({'train': 0.6, 'test': 0.2, 'dev': 0.2})

        assert res['train'].utterances.keys() == {'utt-1', 'utt-3'}
        assert res['test'].utterances.keys() == {'utt-3', 'utt-4'}
        assert res['dev'].utterances.keys() == {'utt-5', 'utt-6'}

        split_mock.assert_called_with(
            {
                'utt-1': {'music': 7, 'speech': 7},
                'utt-2': {'music': 7, 'speech': 7},
                'utt-3': {'music': 1, 'speech': 4},
                'utt-4': {'music': 7, 'speech': 7},
                'utt-5': {'speech': 7},
                'utt-6': {'music': 7, 'speech': 7},
                'utt-7': {'music': 5, 'speech': 6},
                'utt-8': {'music': 10},
            },
            {'train': 0.6, 'test': 0.2, 'dev': 0.2},
            seed=mock.ANY
        )

    @mock.patch('audiomate.corpus.subset.utils.get_identifiers_splitted_by_weights')
    def test_split_by_label_duration_passes_seed(self, split_mock, splitter):
        splitter.split_by_label_duration(
            {'train': 0.5, 'test': 0.5},
            separate_issuers=False
        )

        r = random.Random(INITIAL_SEED)

        split_mock.assert_called_with(
            mock.ANY,
            mock.ANY,
            seed=r.random()
        )

    @mock.patch('audiomate.corpus.subset.utils.get_identifiers_splitted_by_weights')
    def test_split_by_label_duration_issuer_separated(self, split_mock, splitter):
        split_mock.return_value = {
            'train': ['spk-1', 'spk-2'],
            'test': ['spk-3']
        }
        res = splitter.split_by_label_duration(
            {'train': 0.5, 'test': 0.5},
            separate_issuers=True
        )

        assert res['train'].utterances.keys() == {
            'utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'
        }
        assert res['test'].utterances.keys() == {
            'utt-6', 'utt-7', 'utt-8'
        }

        split_mock.assert_called_with(
            {
                'spk-1': {'music': 14, 'speech': 14},
                'spk-2': {'music': 8, 'speech': 18},
                'spk-3': {'music': 22, 'speech': 13},
            },
            {'train': 0.5, 'test': 0.5},
            seed=mock.ANY
        )

    @mock.patch('audiomate.corpus.subset.utils.get_identifiers_splitted_by_weights')
    def test_split_by_label_duration_passes_issuer_separated_seed(self, split_mock, splitter):
        splitter.split_by_label_duration(
            {'train': 0.5, 'test': 0.5},
            separate_issuers=True
        )

        r = random.Random(INITIAL_SEED)

        split_mock.assert_called_with(
            mock.ANY,
            mock.ANY,
            seed=r.random()
        )
