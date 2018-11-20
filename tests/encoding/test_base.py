import os

import numpy as np
from audiomate import encoding

from tests import resources


class EncoderMock(encoding.Encoder):

    def encode_utterance(self, utterance, corpus=None):
        return np.array([1, 2, 3])


class TestEncoder:

    def test_encode_corpus(self, tmpdir):
        ds = resources.create_single_label_corpus()
        target_path = os.path.join(tmpdir.strpath, 'data.hdf5')

        encoder = EncoderMock()
        container = encoder.encode_corpus(ds, target_path)

        with container as ct:
            assert ct.path == target_path
            assert set(ct.keys()) == set(ds.utterances.keys())

            for utterance_idx in ds.utterances.keys():
                assert np.array_equal(ct.get(utterance_idx, mem_map=False), np.array([1, 2, 3]))
