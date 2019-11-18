from audiomate import tracks

from bench import resources


def run(utts):
    for utt in utts:
        utt.read_samples()


def test_utt_read_samples(benchmark):
    utts = []

    wav_path = resources.get_test_resource_path(('wav_files', 'med_len.wav'))
    track = tracks.FileTrack('idx', wav_path)
    utts.append(tracks.Utterance('uidx', track))
    utts.append(tracks.Utterance('uidx', track, start=2.8))
    utts.append(tracks.Utterance('uidx', track, end=10.2))
    utts.append(tracks.Utterance('uidx', track, start=2.4, end=9.8))

    mp3_path = resources.get_test_resource_path(('audio_formats', 'mp3_2_44_1k_16b.mp3'))
    track = tracks.FileTrack('idx', mp3_path)
    utts.append(tracks.Utterance('uidx', track))
    utts.append(tracks.Utterance('uidx', track, start=2.8))
    utts.append(tracks.Utterance('uidx', track, end=4.9))
    utts.append(tracks.Utterance('uidx', track, start=0.4, end=4.8))

    benchmark(run, utts)
