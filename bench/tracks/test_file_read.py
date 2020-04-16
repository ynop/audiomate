from audiomate import tracks

from bench import resources


def run(track):
    for _ in range(300):
        track.read_samples()


def test_read_samples(benchmark):
    wav_path = resources.get_test_resource_path(('wav_files', 'med_len.wav'))
    track = tracks.FileTrack('idx', wav_path)
    benchmark(run, track)
