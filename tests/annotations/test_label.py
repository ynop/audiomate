import numpy as np
import librosa

from audiomate import tracks
from audiomate import annotations
from audiomate import issuers

from tests import resources


class TestLabel:

    def test_label_creation(self):
        a = annotations.Label('value', 6.2, 8.9)

        assert a.value == 'value'
        assert a.start == 6.2
        assert a.end == 8.9
        assert len(a.meta) == 0

    def test_label_creation_with_info(self):
        a = annotations.Label('value', 6.2, 8.9, meta={'something': 2})

        assert a.value == 'value'
        assert a.start == 6.2
        assert a.end == 8.9
        assert len(a.meta) == 1
        assert a.meta['something'] == 2

    def test_lt_start_time_considered_first(self):
        a = annotations.Label('some label', 1.0, 2.0)
        b = annotations.Label('some label', 1.1, 2.0)

        assert a < b

    def test_lt_end_time_considered_second(self):
        a = annotations.Label('some label', 1.0, 1.9)
        b = annotations.Label('some label', 1.0, 2.0)

        assert a < b

    def test_lt_end_time_properly_handles_custom_infinity(self):
        a = annotations.Label('some label', 1.0, 1.9)
        b = annotations.Label('some label', 1.0, -1)

        assert a < b

    def test_lt_value_considered_third(self):
        a = annotations.Label('some label a', 1.0, 2.0)
        b = annotations.Label('some label b', 1.0, 2.0)

        assert a < b

    def test_lt_value_ignores_capitalization(self):
        a = annotations.Label('some label A', 1.0, 2.0)
        b = annotations.Label('some label a', 1.0, 2.0)

        assert not a < b  # not == because == tests different method
        assert not a > b  # not == because == tests different method

    def test_eq_ignores_capitalization(self):
        a = annotations.Label('some label A', 1.0, 2.0)
        b = annotations.Label('some label a', 1.0, 2.0)

        assert a == b

    def test_eq_ignores_label_list_relation(self):
        a = annotations.Label('some label A', 1.0, 2.0)
        b = annotations.Label('some label a', 1.0, 2.0)

        al = annotations.LabelList(idx='one', labels=[a])
        bl = annotations.LabelList(idx='another', labels=[b])

        assert a.label_list == al
        assert b.label_list == bl
        assert a == b

    def test_lt_ignores_label_list_relation(self):
        a = annotations.Label('some label A', 1.0, 2.0)
        b = annotations.Label('some label a', 1.0, 2.0)

        al = annotations.LabelList(idx='one', labels=[a])
        bl = annotations.LabelList(idx='another', labels=[b])

        assert a.label_list == al
        assert b.label_list == bl
        assert not a < b
        assert not a > b

    def test_read_samples(self):
        path = resources.sample_wav_file('wav_1.wav')
        track = tracks.FileTrack('wav', path)
        issuer = issuers.Issuer('toni')
        utt = tracks.Utterance('t', track, issuer=issuer, start=1.0, end=2.30)

        l1 = annotations.Label('a', 0.15, 0.448)
        l2 = annotations.Label('a', 0.5, 0.73)
        ll = annotations.LabelList(labels=[l1, l2])

        utt.set_label_list(ll)

        expected, __ = librosa.core.load(
            path,
            sr=None,
            offset=1.15,
            duration=0.298
        )
        assert np.array_equal(l1.read_samples(), expected)

        expected, __ = librosa.core.load(
            path,
            sr=None,
            offset=1.5,
            duration=0.23
        )
        assert np.array_equal(l2.read_samples(), expected)

    def test_read_samples_no_utterance_and_label_end(self):
        path = resources.sample_wav_file('wav_1.wav')
        track = tracks.FileTrack('wav', path)
        issuer = issuers.Issuer('toni')
        utt = tracks.Utterance('idx', track, issuer=issuer, start=1.0, end=-1)

        l1 = annotations.Label('a', 0.15, 0.448)
        l2 = annotations.Label('a', 0.5, -1)
        ll = annotations.LabelList(labels=[l1, l2])

        utt.set_label_list(ll)

        expected, __ = librosa.core.load(
            path,
            sr=None,
            offset=1.15,
            duration=0.298
        )
        assert np.array_equal(l1.read_samples(), expected)

        expected, __ = librosa.core.load(path, sr=None, offset=1.5)
        assert np.array_equal(l2.read_samples(), expected)

    def test_start_abs(self):
        label = annotations.Label('a', 2, 5)
        ll = annotations.LabelList(labels=[label])
        tracks.Utterance('utt-1', None, start=1, end=19, label_lists=[ll])

        assert label.start_abs == 3

    def test_start_abs_no_utterance(self):
        label = annotations.Label('a', 2, 5)

        assert label.start_abs == 2

    def test_end_abs(self):
        label = annotations.Label('a', 2, 5)
        ll = annotations.LabelList(labels=[label])
        tracks.Utterance('utt-1', None, start=1, end=19, label_lists=[ll])

        assert label.end_abs == 6

    def test_end_abs_no_utterance(self):
        label = annotations.Label('a', 2, 5)

        assert label.end_abs == 5

    def test_duration(self):
        label = annotations.Label('a', 2, 5)

        assert label.duration == 3

    def test_tokenized(self):
        label = annotations.Label('wo wie was warum  weshalb')

        assert label.tokenized(delimiter=' ') == [
            'wo', 'wie', 'was', 'warum', 'weshalb'
        ]

    def test_tokenized_returns_empty_list_for_empty_label_value(self):
        label = annotations.Label('  ')

        assert label.tokenized(delimiter=' ') == []

    def test_tokenized_with_other_delimiter(self):
        label = annotations.Label('wie,kann, ich, dass machen')

        assert label.tokenized(delimiter=',') == ['wie', 'kann', 'ich', 'dass machen']
