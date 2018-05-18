import unittest
import os

from audiomate.corpus import io
from audiomate.corpus import assets
from tests import resources


class MusanReaderTest(unittest.TestCase):
    def setUp(self):
        self.reader = io.MusanReader()
        self.test_path = resources.sample_corpus_path('musan')

    def test_load_files(self):
        ds = self.reader.load(self.test_path)

        fma = os.path.join(self.test_path, 'music', 'fma')
        free_sound = os.path.join(self.test_path, 'noise', 'free-sound')
        librivox = os.path.join(self.test_path, 'speech', 'librivox')

        assert ds.num_files == 5

        assert ds.files['music-fma-0000'].idx == 'music-fma-0000'
        assert ds.files['music-fma-0000'].path == os.path.join(fma, 'music-fma-0000.wav')

        assert ds.files['noise-free-sound-0000'].idx == 'noise-free-sound-0000'
        assert ds.files['noise-free-sound-0000'].path == os.path.join(free_sound, 'noise-free-sound-0000.wav')
        assert ds.files['noise-free-sound-0001'].idx == 'noise-free-sound-0001'
        assert ds.files['noise-free-sound-0001'].path == os.path.join(free_sound, 'noise-free-sound-0001.wav')

        assert ds.files['speech-librivox-0000'].idx == 'speech-librivox-0000'
        assert ds.files['speech-librivox-0000'].path == os.path.join(librivox, 'speech-librivox-0000.wav')
        assert ds.files['speech-librivox-0001'].idx == 'speech-librivox-0001'
        assert ds.files['speech-librivox-0001'].path == os.path.join(librivox, 'speech-librivox-0001.wav')

    def test_load_issuers(self):
        ds = self.reader.load(self.test_path)

        assert ds.num_issuers == 3

        assert 'speech-librivox-0000' in ds.issuers.keys()
        assert type(ds.issuers['speech-librivox-0000']) == assets.Speaker
        assert ds.issuers['speech-librivox-0000'].idx == 'speech-librivox-0000'
        assert ds.issuers['speech-librivox-0000'].gender == assets.Gender.MALE

        assert 'speech-librivox-0001' in ds.issuers.keys()
        assert type(ds.issuers['speech-librivox-0001']) == assets.Speaker
        assert ds.issuers['speech-librivox-0001'].idx == 'speech-librivox-0001'
        assert ds.issuers['speech-librivox-0001'].gender == assets.Gender.FEMALE

        assert 'Quiet_Music_for_Tiny_Robots' in ds.issuers.keys()
        assert type(ds.issuers['Quiet_Music_for_Tiny_Robots']) == assets.Artist
        assert ds.issuers['Quiet_Music_for_Tiny_Robots'].idx == 'Quiet_Music_for_Tiny_Robots'
        assert ds.issuers['Quiet_Music_for_Tiny_Robots'].name == 'Quiet_Music_for_Tiny_Robots'

    def test_load_utterances(self):
        ds = self.reader.load(self.test_path)

        assert ds.num_utterances == 5

        assert ds.utterances['music-fma-0000'].idx == 'music-fma-0000'
        assert ds.utterances['music-fma-0000'].file.idx == 'music-fma-0000'
        assert ds.utterances['music-fma-0000'].issuer.idx == 'Quiet_Music_for_Tiny_Robots'
        assert ds.utterances['music-fma-0000'].start == 0
        assert ds.utterances['music-fma-0000'].end == -1

        assert ds.utterances['noise-free-sound-0000'].idx == 'noise-free-sound-0000'
        assert ds.utterances['noise-free-sound-0000'].file.idx == 'noise-free-sound-0000'
        assert ds.utterances['noise-free-sound-0000'].issuer is None
        assert ds.utterances['noise-free-sound-0000'].start == 0
        assert ds.utterances['noise-free-sound-0000'].end == -1

        assert ds.utterances['noise-free-sound-0001'].idx == 'noise-free-sound-0001'
        assert ds.utterances['noise-free-sound-0001'].file.idx == 'noise-free-sound-0001'
        assert ds.utterances['noise-free-sound-0001'].issuer is None
        assert ds.utterances['noise-free-sound-0001'].start == 0
        assert ds.utterances['noise-free-sound-0001'].end == -1

        assert ds.utterances['speech-librivox-0000'].idx == 'speech-librivox-0000'
        assert ds.utterances['speech-librivox-0000'].file.idx == 'speech-librivox-0000'
        assert ds.utterances['speech-librivox-0000'].issuer.idx == 'speech-librivox-0000'
        assert ds.utterances['speech-librivox-0000'].start == 0
        assert ds.utterances['speech-librivox-0000'].end == -1

        assert ds.utterances['speech-librivox-0001'].idx == 'speech-librivox-0001'
        assert ds.utterances['speech-librivox-0001'].file.idx == 'speech-librivox-0001'
        assert ds.utterances['speech-librivox-0001'].issuer.idx == 'speech-librivox-0001'
        assert ds.utterances['speech-librivox-0001'].start == 0
        assert ds.utterances['speech-librivox-0001'].end == -1

    def test_load_label_lists(self):
        ds = self.reader.load(self.test_path)

        utt_1 = ds.utterances['music-fma-0000']
        utt_2 = ds.utterances['noise-free-sound-0000']
        utt_3 = ds.utterances['noise-free-sound-0001']

        assert 'audio_type' in utt_1.label_lists.keys()
        assert 'audio_type' in utt_2.label_lists.keys()
        assert 'audio_type' in utt_3.label_lists.keys()

        assert len(utt_1.label_lists['audio_type'].labels) == 1
        assert len(utt_2.label_lists['audio_type'].labels) == 1
        assert len(utt_3.label_lists['audio_type'].labels) == 1

        assert utt_1.label_lists['audio_type'].labels[0].value == 'music'
        assert utt_2.label_lists['audio_type'].labels[0].value == 'noise'
        assert utt_3.label_lists['audio_type'].labels[0].value == 'noise'
