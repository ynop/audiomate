import numpy as np

from audiomate import tracks
from audiomate import annotations
from audiomate import encoding

import pytest


class TestTokenOrdinalEncoder:

    def test_encode_utterance_with_single_label(self):
        ll = annotations.LabelList(idx='go', labels=[
            annotations.Label('a c b')
        ])
        utt = tracks.Utterance('utt-1', None, label_lists=ll)

        encoder = encoding.TokenOrdinalEncoder('go', ['a', 'b', 'c'])
        encoded = encoder.encode_utterance(utt)

        assert np.array_equal(encoded, [0, 2, 1])

    def test_encode_utterance_with_multiple_non_overlapping_labels(self):
        ll = annotations.LabelList(idx='go', labels=[
            annotations.Label('a c b', start=0, end=5),
            annotations.Label('c b b', start=5, end=9.4),
            annotations.Label('a a a', start=9.5, end=10.2)
        ])
        utt = tracks.Utterance('utt-1', None, label_lists=ll)

        encoder = encoding.TokenOrdinalEncoder('go', ['a', 'b', 'c'])
        encoded = encoder.encode_utterance(utt)

        assert np.array_equal(encoded, [0, 2, 1, 2, 1, 1, 0, 0, 0])

    def test_encode_utterance_with_overlapping_labels_raises_error(self):
        ll = annotations.LabelList(idx='go', labels=[
            annotations.Label('a c b', start=0, end=5),
            annotations.Label('c b b', start=2, end=9.4),
            annotations.Label('a a a', start=9.5, end=10.2)
        ])
        utt = tracks.Utterance('utt-1', None, label_lists=ll)

        encoder = encoding.TokenOrdinalEncoder('go', ['a', 'b', 'c'])

        with pytest.raises(ValueError):
            encoder.encode_utterance(utt)

    def test_encode_utterance_with_missing_token_raises_error(self):
        ll = annotations.LabelList(idx='go', labels=[
            annotations.Label('a c b unknown')
        ])
        utt = tracks.Utterance('utt-1', None, label_lists=ll)

        encoder = encoding.TokenOrdinalEncoder('go', ['a', 'b', 'c'])

        with pytest.raises(ValueError):
            encoder.encode_utterance(utt)

    def test_encode_utterance_with_non_existing_label_list_raises_error(self):
        ll = annotations.LabelList(idx='go', labels=[
            annotations.Label('a c b unknown')
        ])
        utt = tracks.Utterance('utt-1', None, label_lists=ll)

        encoder = encoding.TokenOrdinalEncoder('not_existing', ['a', 'b', 'c'])

        with pytest.raises(ValueError):
            encoder.encode_utterance(utt)

    def test_encode_utterance_with_custom_delimiter(self):
        ll = annotations.LabelList(idx='go', labels=[
            annotations.Label('a, c , b, b')
        ])
        utt = tracks.Utterance('utt-1', None, label_lists=ll)

        encoder = encoding.TokenOrdinalEncoder('go', ['a', 'b', 'c'], token_delimiter=',')
        encoded = encoder.encode_utterance(utt)

        assert np.array_equal(encoded, [0, 2, 1, 1])
