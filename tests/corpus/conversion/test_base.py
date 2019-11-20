import os

import pytest
from tests import resources

from audiomate.corpus.conversion import base


class DummyAudioFileConverter(base.AudioFileConverter):

    def __init__(self, a_ret, sampling_rate=16000, separate_file_per_utterance=False,
                 force_conversion=False):
        super(DummyAudioFileConverter, self).__init__(
            sampling_rate,
            separate_file_per_utterance,
            force_conversion
        )
        self.a_ret = a_ret
        self.a_log = []
        self.b_log = []

    def _file_extension(self):
        return 'wav'

    def _does_utt_match_target_format(self, utterance):
        self.a_log.append(utterance.idx)
        return self.a_ret[utterance.idx]

    def _convert_files(self, files):
        self.b_log.append(files)


@pytest.fixture
def ds():
    ds = resources.create_dataset()

    file_1_path = resources.sample_wav_file('wav_1.wav')
    file_2_path = resources.get_resource_path(('audio_formats', 'mp3_2_44_1k_16b.mp3'))
    file_3_path = resources.get_resource_path(('audio_formats', 'flac_1_16k_16b.flac'))
    file_4_path = resources.sample_wav_file('wav_4.wav')

    ds.tracks['wav-1'].path = file_1_path
    ds.tracks['wav_2'].path = file_2_path
    ds.tracks['wav_3'].path = file_3_path
    ds.tracks['wav_4'].path = file_4_path

    return ds


class TestAudioFileConverter:

    def test_convert(self, tmp_path, ds):
        c = DummyAudioFileConverter({
            'utt-1': True, 'utt-2': False, 'utt-3': False, 'utt-4': False, 'utt-5': True
        })
        res = c.convert(ds, str(tmp_path))

        assert sorted(c.a_log) == sorted(ds.utterances.keys())
        assert len(c.b_log) == 1

        assert sorted(c.b_log[0]) == sorted([
            (
                ds.tracks['wav_2'].path,
                0,
                float('inf'),
                os.path.join(str(tmp_path), 'wav_2.wav')
            ),
            (
                ds.tracks['wav_3'].path,
                0,
                float('inf'),
                os.path.join(str(tmp_path), 'wav_3.wav')
            )
        ])

        assert set(res.utterances.keys()) == set(ds.utterances.keys())
        assert set(res.tracks.keys()) == set(ds.tracks.keys())
        assert set(res.subviews.keys()) == set(ds.subviews.keys())

        assert res.utterances['utt-2'].track.path == os.path.join(str(tmp_path), 'wav_2.wav')

    def test_convert_separate_file_per_utterance(self, tmp_path, ds):
        c = DummyAudioFileConverter({
            'utt-1': True, 'utt-2': False, 'utt-3': False, 'utt-4': False, 'utt-5': True
        }, separate_file_per_utterance=True)
        res = c.convert(ds, str(tmp_path))

        assert sorted(c.a_log) == ['utt-1', 'utt-2', 'utt-5']
        assert len(c.b_log) == 1

        assert sorted(c.b_log[0]) == sorted([
            (
                ds.tracks['wav_2'].path,
                0,
                float('inf'),
                os.path.join(str(tmp_path), 'utt-2.wav')
            ),
            (
                ds.tracks['wav_3'].path,
                0.0,
                1.5,
                os.path.join(str(tmp_path), 'utt-3.wav')
            ),
            (
                ds.tracks['wav_3'].path,
                1.5,
                2.5,
                os.path.join(str(tmp_path), 'utt-4.wav')
            ),
        ])

        assert set(res.utterances.keys()) == set(ds.utterances.keys())
        assert set(res.tracks.keys()) == {'wav-1', 'utt-2', 'utt-3', 'utt-4', 'wav_4'}
        assert set(res.subviews.keys()) == set(ds.subviews.keys())

        assert res.utterances['utt-2'].track.path == os.path.join(str(tmp_path), 'utt-2.wav')

    def test_convert_separate_file_per_utterance_and_force(self, tmp_path, ds):
        c = DummyAudioFileConverter({
            'utt-1': True, 'utt-2': False, 'utt-3': False, 'utt-4': False, 'utt-5': True
        }, separate_file_per_utterance=True, force_conversion=True)
        res = c.convert(ds, str(tmp_path))

        assert sorted(c.a_log) == []
        assert len(c.b_log) == 1

        assert sorted(c.b_log[0]) == sorted([
            (
                ds.tracks['wav-1'].path,
                0,
                float('inf'),
                os.path.join(str(tmp_path), 'utt-1.wav')
            ),
            (
                ds.tracks['wav_2'].path,
                0,
                float('inf'),
                os.path.join(str(tmp_path), 'utt-2.wav')
            ),
            (
                ds.tracks['wav_3'].path,
                0.0,
                1.5,
                os.path.join(str(tmp_path), 'utt-3.wav')
            ),
            (
                ds.tracks['wav_3'].path,
                1.5,
                2.5,
                os.path.join(str(tmp_path), 'utt-4.wav')
            ),
            (
                ds.tracks['wav_4'].path,
                0,
                float('inf'),
                os.path.join(str(tmp_path), 'utt-5.wav')
            ),
        ])

        assert set(res.utterances.keys()) == set(ds.utterances.keys())
        assert set(res.tracks.keys()) == {'utt-1', 'utt-2', 'utt-3', 'utt-4', 'utt-5'}
        assert set(res.subviews.keys()) == set(ds.subviews.keys())

        assert res.utterances['utt-2'].track.path == os.path.join(str(tmp_path), 'utt-2.wav')

    def test_convert_force(self, tmp_path, ds):
        c = DummyAudioFileConverter({
            'utt-1': True, 'utt-2': False, 'utt-3': False, 'utt-4': False, 'utt-5': True
        }, force_conversion=True)
        res = c.convert(ds, str(tmp_path))

        assert sorted(c.a_log) == []
        assert len(c.b_log) == 1

        assert sorted(c.b_log[0]) == sorted([
            (
                ds.tracks['wav-1'].path,
                0,
                float('inf'),
                os.path.join(str(tmp_path), 'wav-1.wav')
            ),
            (
                ds.tracks['wav_2'].path,
                0,
                float('inf'),
                os.path.join(str(tmp_path), 'wav_2.wav')
            ),
            (
                ds.tracks['wav_3'].path,
                0,
                float('inf'),
                os.path.join(str(tmp_path), 'wav_3.wav')
            ),
            (
                ds.tracks['wav_4'].path,
                0,
                float('inf'),
                os.path.join(str(tmp_path), 'wav_4.wav')
            ),
        ])

        assert set(res.utterances.keys()) == set(ds.utterances.keys())
        assert set(res.tracks.keys()) == set(ds.tracks.keys())
        assert set(res.subviews.keys()) == set(ds.subviews.keys())

        assert res.utterances['utt-2'].track.path == os.path.join(str(tmp_path), 'wav_2.wav')
