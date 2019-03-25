import numpy as np
import librosa
import pytest

from audiomate import tracks
from audiomate import annotations
from audiomate import issuers

from tests import resources


class TestUtterance:

    def setup_method(self):
        self.ll_1 = annotations.LabelList(idx='alpha', labels=[
            annotations.Label('a', 3.2, 4.5),
            annotations.Label('b', 5.1, 8.9),
            annotations.Label('c', 7.2, 10.5),
            annotations.Label('d', 10.5, 14),
            annotations.Label('d', 15, 18)
        ])

        self.ll_2 = annotations.LabelList(idx='bravo', labels=[
            annotations.Label('a', 1.0, 4.2),
            annotations.Label('e', 4.2, 7.9),
            annotations.Label('c', 7.2, 10.5),
            annotations.Label('f', 10.5, 14),
            annotations.Label('d', 15, 17.3)
        ])

        self.ll_duplicate_idx = annotations.LabelList(idx='charlie', labels=[
            annotations.Label('t', 1.0, 4.2),
            annotations.Label('h', 4.2, 7.9)
        ])

        self.ll_3 = annotations.LabelList(idx='charlie', labels=[
            annotations.Label('a', 1.0, 4.2),
            annotations.Label('g', 4.2, 7.9)
        ])

        self.track = tracks.FileTrack('wav', resources.sample_wav_file('wav_1.wav'))
        self.issuer = issuers.Issuer('toni')
        self.utt = tracks.Utterance(
            'test',
            self.track,
            issuer=self.issuer,
            start=1.25,
            end=1.30,
            label_lists=[
                self.ll_1,
                self.ll_2,
                self.ll_duplicate_idx,
                self.ll_3
            ]
        )

    def test_end_abs(self):
        assert self.utt.end_abs == 1.30

    def test_end_abs_end_of_track(self):
        utt = tracks.Utterance('utt', self.track, start=0.3)
        assert utt.end_abs == pytest.approx(2.5951875)

    def test_duration(self):
        assert self.utt.duration == pytest.approx(0.05)

    def test_duration_end_of_track(self):
        utt = tracks.Utterance('utt', self.track, start=0.3)
        assert utt.duration == pytest.approx(2.2951875)

    def test_duration_longer_than_track(self):
        utt = tracks.Utterance('utt', self.track, start=0.3, end=255.0)
        assert utt.duration == pytest.approx(254.7)

    def test_issuer_relation_on_creation(self):
        assert self.utt.issuer == self.issuer
        assert self.utt in self.issuer.utterances

    def test_set_label_list(self):
        ll_4 = annotations.LabelList(idx='delta', labels=[
            annotations.Label('y', 0.0, 3.3),
            annotations.Label('t', 3.8, 7.9)
        ])

        self.utt.set_label_list(ll_4)

        assert len(self.utt.label_lists) == 4
        assert self.utt.label_lists['alpha'] == self.ll_1
        assert self.utt.label_lists['bravo'] == self.ll_2
        assert self.utt.label_lists['charlie'] == self.ll_3
        assert self.utt.label_lists['delta'] == ll_4
        assert self.ll_1.utterance == self.utt
        assert self.ll_2.utterance == self.utt
        assert self.ll_3.utterance == self.utt
        assert ll_4.utterance == self.utt

    def test_label_values(self):
        assert self.utt.all_label_values() == {'a', 'b', 'c', 'd', 'e', 'f', 'g'}

    def test_label_values_with_idx_restriction(self):
        all_values = self.utt.all_label_values(label_list_ids=['bravo', 'charlie'])
        assert all_values == {'a', 'c', 'd', 'e', 'f', 'g'}

    def test_label_count(self):
        assert self.utt.label_count() == {'a': 3, 'b': 1, 'c': 2, 'd': 3, 'e': 1, 'f': 1, 'g': 1}

    def test_label_count_with_idx_restriction(self):
        count = self.utt.label_count(label_list_ids=['bravo', 'charlie'])
        assert count == {'a': 2, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 1}

    def test_label_duration(self):
        duration = self.utt.label_total_duration(label_list_ids=['alpha', 'charlie'])
        assert duration['a'] == pytest.approx(4.5)
        assert duration['b'] == pytest.approx(3.8)
        assert duration['c'] == pytest.approx(3.3)
        assert duration['d'] == pytest.approx(6.5)
        assert duration['g'] == pytest.approx(3.7)

    def test_read_samples(self):
        expected, __ = librosa.core.load(self.track.path, sr=None, offset=1.25, duration=0.05)
        assert np.array_equal(self.utt.read_samples(), expected)

    def test_read_samples_with_offset(self):
        expected, __ = librosa.core.load(self.track.path, sr=None, offset=1.27, duration=0.03)
        assert np.array_equal(self.utt.read_samples(offset=0.02), expected)

    def test_read_samples_with_duration(self):
        expected, __ = librosa.core.load(self.track.path, sr=None, offset=1.26, duration=0.03)
        assert np.array_equal(self.utt.read_samples(offset=0.01, duration=0.03), expected)

    def test_read_samples_with_duration_no_end_set(self):
        self.utt.end = float('inf')

        expected, __ = librosa.core.load(self.track.path, sr=None, offset=1.26, duration=0.03)
        assert np.array_equal(self.utt.read_samples(offset=0.01, duration=0.03), expected)

    def test_num_samples(self):
        assert self.utt.num_samples() == 800

    def test_num_samples_non_native_sr(self):
        assert self.utt.num_samples(sr=12000) == 600

    def test_num_samples_matches_read_samples(self):
        assert self.utt.read_samples().shape[0] == self.utt.num_samples()
        assert self.utt.read_samples(sr=11255).shape[0] == self.utt.num_samples(sr=11255)

    def test_split(self):
        ll_1 = annotations.LabelList('phones', labels=[
            annotations.Label('alpha', start=0.0, end=30.0)
        ])
        ll_2 = annotations.LabelList('words', labels=[
            annotations.Label('b', start=0.0, end=30.0)
        ])
        utt = tracks.Utterance('utt-1', 'track-x', start=0.0, end=40.0, label_lists=[ll_1, ll_2])

        res = utt.split([14.0, 29.5])

        assert len(res) == 3

        assert res[0].start == 0.0
        assert res[0].end == 14.0
        assert 'phones' in res[0].label_lists.keys()
        assert 'words' in res[0].label_lists.keys()

        assert res[1].start == 14.0
        assert res[1].end == 29.5
        assert 'phones' in res[1].label_lists.keys()
        assert 'words' in res[1].label_lists.keys()

        assert res[2].start == 29.5
        assert res[2].end == 40.0
        assert 'phones' in res[2].label_lists.keys()
        assert 'words' in res[2].label_lists.keys()

    def test_split_endless(self):
        utt = tracks.Utterance('utt-1', None, start=0.0)
        res = utt.split([24.5])

        assert len(res) == 2
        assert res[0].start == 0.0
        assert res[0].end == 24.5
        assert res[1].start == 24.5
        assert res[1].end == float('inf')

    def test_split_sets_track(self):
        file = tracks.FileTrack('file-1', '/some/path')
        utt = tracks.Utterance('utt-1', file, start=0.0, end=10.0)
        res = utt.split([5.2])

        assert len(res) == 2
        assert res[0].track == file
        assert res[1].track == file

    def test_split_sets_issuer(self):
        issuer = issuers.Speaker('spk-1')
        utt = tracks.Utterance('utt-1', None, issuer=issuer, start=0.0, end=10.0)
        res = utt.split([5.2])

        assert len(res) == 2
        assert res[0].issuer == issuer
        assert res[1].issuer == issuer

    def test_split_without_cutting_points_raises_error(self):
        utt = tracks.Utterance('utt-1', None, start=0.0)

        with pytest.raises(ValueError):
            utt.split([])

    def test_split_with_cutting_point_after_end_returns_one_utt(self):
        utt = tracks.Utterance('utt-1', None, start=4.0, end=20.0)
        res = utt.split([24.5])

        assert len(res) == 1
        assert res[0].start == 4.0
        assert res[0].end == 20.0

    def test_split_when_utt_start_is_not_zero(self):
        utt = tracks.Utterance('utt-1', None, start=6.0, end=20.0)
        res = utt.split([3.0])

        assert len(res) == 2
        assert res[0].start == 6.0
        assert res[0].end == 9.0
        assert res[1].start == 9.0
        assert res[1].end == 20.

    def test_split_track_relative(self):
        utt = tracks.Utterance('utt-1', None, start=6.0, end=20.0)
        res = utt.split([8.0], track_relative=True)

        assert len(res) == 2
        assert res[0].start == 6.0
        assert res[0].end == 8.0
        assert res[1].start == 8.0
        assert res[1].end == 20.00

    def test_split_utt_relative(self):
        utt = tracks.Utterance('utt-1', None, start=6.0, end=20.0)
        res = utt.split([8.0], track_relative=False)

        assert len(res) == 2
        assert res[0].start == 6.0
        assert res[0].end == 14.0
        assert res[1].start == 14.0
        assert res[1].end == 20.0

    def test_split_utt_relative_with_labels(self):
        ll_1 = annotations.LabelList('phones', labels=[
            annotations.Label('alpha', start=0.0, end=30.0)
        ])
        ll_2 = annotations.LabelList('words', labels=[
            annotations.Label('b', start=8.0, end=30.0)
        ])
        utt = tracks.Utterance('utt-1', 'file-x', start=10.0, end=40.0, label_lists=[ll_1, ll_2])

        res = utt.split([14.0], track_relative=False)

        assert len(res) == 2

        assert res[0].start == 10.0
        assert res[0].end == 24.0
        assert res[0].label_lists['phones'] == annotations.LabelList(idx='phones', labels=[
            annotations.Label('alpha', 0.0, 14.0)
        ])
        assert res[0].label_lists['words'] == annotations.LabelList(idx='words', labels=[
            annotations.Label('b', 8.0, 14.0)
        ])

        assert res[1].start == 24.0
        assert res[1].end == 40.0
        assert res[1].label_lists['phones'] == annotations.LabelList(idx='phones', labels=[
            annotations.Label('alpha', 0.0, 16.0)
        ])
        assert res[1].label_lists['words'] == annotations.LabelList(idx='words', labels=[
            annotations.Label('b', 0.0, 16.0)
        ])

    def test_split_with_overlap(self):
        ll_1 = annotations.LabelList('phones', labels=[
            annotations.Label('alpha', start=0.0, end=30.0),
            annotations.Label('bravo', start=20.0, end=42.0)
        ])
        ll_2 = annotations.LabelList('words', labels=[
            annotations.Label('b', start=8.0, end=30.0)
        ])
        utt = tracks.Utterance('utt-1', 'file-x', start=10.0, end=55.0, label_lists=[ll_1, ll_2])

        res = utt.split([12.0, 24.0], track_relative=False, overlap=2.0)

        assert len(res) == 3

        assert res[0].start == 10.0
        assert res[0].end == 24.0
        assert res[0].label_lists['phones'] == annotations.LabelList(idx='phones', labels=[
            annotations.Label('alpha', 0.0, 14.0)
        ])
        assert res[0].label_lists['words'] == annotations.LabelList(idx='words', labels=[
            annotations.Label('b', 8.0, 14.0)
        ])

        print(res[1].label_lists['phones'].labels)

        assert res[1].start == 20.0
        assert res[1].end == 36.0
        assert res[1].label_lists['phones'] == annotations.LabelList(idx='phones', labels=[
            annotations.Label('alpha', 0.0, 16.0),
            annotations.Label('bravo', 10.0, 16.0),
        ])
        assert res[1].label_lists['words'] == annotations.LabelList(idx='words', labels=[
            annotations.Label('b', 0.0, 16.0)
        ])

        assert res[2].start == 32.0
        assert res[2].end == 55.0
        assert res[2].label_lists['phones'] == annotations.LabelList(idx='phones', labels=[
            annotations.Label('alpha', 0.0, 8.0),
            annotations.Label('bravo', 0.0, 20.0),
        ])
        assert res[2].label_lists['words'] == annotations.LabelList(idx='words', labels=[
            annotations.Label('b', 0.0, 8.0)
        ])
