import os
import shutil

import numpy as np

import audiomate
from audiomate.utils import audio


def run(path):
    audiomate.Corpus.load(path, reader='tuda')


def generate_corpus(path):
    file_template = """
                    <?xml version = "1.0" encoding = "utf-8"?>
                    <recording>
                    <speaker_id>SPEAKERIDX</speaker_id>
                    <rate>16000</rate>
                    <angle>-10,0267614147894</angle>
                    <gender>male</gender>
                    <ageclass>31-40</ageclass>
                    <sentence_id>1</sentence_id>
                    <sentence>In der römisch-katholischen Kirche</sentence>
                    <cleaned_sentence>In der römisch katholische</cleaned_sentence>
                    <corpus>WIKI</corpus>
                    <muttersprachler>Ja</muttersprachler>
                    <bundesland>Hessen</bundesland>
                    <sourceurls><url>https://de.wikipedia.org/wiki/Römisches_Reich</url></sourceurls></recording>
                    """

    for part in ('train', 'dev', 'test'):
        part_path = os.path.join(path, part)

        if os.path.isdir(part_path):
            shutil.rmtree(part_path)
        os.makedirs(part_path, exist_ok=True)

        for idx in range(400):
            xml_path = os.path.join(part_path, '{}.xml'.format(idx))
            xml_content = file_template.replace('SPEAKERIDX', str(idx))
            with open(xml_path, 'w') as f:
                f.write(xml_content)

            for wav_idx in range(3):
                wav_path = os.path.join(part_path, '{}_{}.wav'.format(idx, wav_idx))
                audio.write_wav(wav_path, np.array([1, 2, 3]))


def test_load_tuda(benchmark, tmp_path):
    temp_dir = str(tmp_path)
    generate_corpus(temp_dir)

    benchmark(run, temp_dir)
