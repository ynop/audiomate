import numpy as np

from audiomate.processing import pipeline


class TestAvgPool:

    def test_compute(self):
        avg_pooling = pipeline.AvgPool(3)

        data_a = np.array([
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5]
        ])
        chunk_a = pipeline.Chunk(data_a, 0, False)

        out_a = avg_pooling.compute(chunk_a, 16000)

        assert np.allclose(out_a, np.array([
            [2, 2]
        ]))

        data_b = np.array([
            [6, 1],
            [7, 1],
            [8, 1],
            [9, 1],
        ])
        chunk_b = pipeline.Chunk(data_b, 0, True)

        out_b = avg_pooling.compute(chunk_b, 16000)

        assert np.allclose(out_b, np.array([
            [5, 10.0 / 3.0],
            [8, 1]
        ]))

    def test_compute_with_rest(self):
        avg_pooling = pipeline.AvgPool(3)

        data_a = np.array([
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5]
        ])
        chunk_a = pipeline.Chunk(data_a, 0, True)

        out_a = avg_pooling.compute(chunk_a, 16000)

        assert np.allclose(out_a, np.array([
            [2, 2],
            [4.5, 4.5]
        ]))


class TestVarPool:

    def test_compute(self):
        var_pooling = pipeline.VarPool(3)

        data_a = np.array([
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5]
        ])
        chunk_a = pipeline.Chunk(data_a, 0, False)

        out_a = var_pooling.compute(chunk_a, 16000)

        assert np.allclose(out_a, np.array([
            [0.6666666666666666, 0.6666666666666666]
        ]))

        data_b = np.array([
            [6, 1],
            [7, 1],
            [8, 1],
            [9, 1],
        ])
        chunk_b = pipeline.Chunk(data_b, 0, True)

        out_b = var_pooling.compute(chunk_b, 16000)

        assert np.allclose(out_b, np.array([
            [0.6666666666666666, 2.8888888888888893],
            [0.6666666666666666, 0]
        ]))

    def test_compute_with_rest(self):
        var_pooling = pipeline.VarPool(3)

        data_a = np.array([
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5]
        ])
        chunk_a = pipeline.Chunk(data_a, 0, True)

        out_a = var_pooling.compute(chunk_a, 16000)

        assert np.allclose(out_a, np.array([
            [0.6666666666666666, 0.6666666666666666],
            [0.25, 0.25]
        ]))
