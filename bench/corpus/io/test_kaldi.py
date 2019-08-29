import random
import os
import shutil

from bench import resources

from audiomate.corpus import io


def run(corpus, base_path):
    target_path = os.path.join(base_path, 'out')
    shutil.rmtree(target_path, ignore_errors=True)
    os.makedirs(target_path)

    writer = io.KaldiWriter()
    writer.save(corpus, target_path)


def test_kaldi_write(benchmark, tmp_path):
    corpus = resources.generate_corpus(
        200,
        (5, 10),
        (1, 5),
        (0, 6),
        (1, 20),
        random.Random(x=234)
    )

    benchmark(run, corpus, str(tmp_path))
