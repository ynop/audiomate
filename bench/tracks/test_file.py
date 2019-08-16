from audiomate import tracks

from bench import resources


def run(track):
    for i in range(300):
        track.duration


def test_duration(benchmark):
    wav_path = resources.get_test_resource_path(('wav_files', 'med_len.wav'))
    track = tracks.FileTrack('idx', wav_path)
    benchmark(run, track)
