from audiomate.corpus import assets
from audiomate import feeding

import pytest
from tests import resources


class TestDataIterator:

    def test_init_with_utterance_list(self):
        it = feeding.DataIterator(['utt-1', 'utt-2'], [assets.Container('blub')])
        assert set(it.utt_ids) == {'utt-1', 'utt-2'}

    def test_init_with_corpus(self):
        corpus = resources.create_dataset()
        it = feeding.DataIterator(corpus, [assets.Container('blub')])
        assert set(it.utt_ids) == set(corpus.utterances.keys())

    def test_init_throws_error_when_no_container_is_given(self):
        corpus = resources.create_dataset()

        with pytest.raises(ValueError):
            feeding.DataIterator(corpus, [])
