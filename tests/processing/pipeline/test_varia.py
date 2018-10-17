import numpy as np

from audiomate.processing import pipeline


class TestAddContext:

    def test_compute_with_left_frames(self):
        data = np.array([[1, 2], [3, 4], [5, 6]])
        chunk = pipeline.Chunk(data, offset=0, is_last=True)

        step = pipeline.AddContext(left_frames=2, right_frames=0)
        result = step.compute(chunk, 16000)

        assert np.array_equal(result, np.array([
            [0, 0, 0, 0, 1, 2],
            [0, 0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5, 6]
        ]))

    def test_compute_with_right_frames(self):
        data = np.array([[1, 2], [3, 4], [5, 6]])
        chunk = pipeline.Chunk(data, offset=0, is_last=True)

        step = pipeline.AddContext(left_frames=0, right_frames=2)
        result = step.compute(chunk, 16000)

        assert np.array_equal(result, np.array([
            [1, 2, 3, 4, 5, 6],
            [3, 4, 5, 6, 0, 0],
            [5, 6, 0, 0, 0, 0]
        ]))

    def test_compute_with_left_and_right_frames(self):
        data = np.array([[1, 2], [3, 4], [5, 6]])
        chunk = pipeline.Chunk(data, offset=0, is_last=True)

        step = pipeline.AddContext(left_frames=2, right_frames=2)
        result = step.compute(chunk, 16000)

        assert np.array_equal(result, np.array([
            [0, 0, 0, 0, 1, 2, 3, 4, 5, 6],
            [0, 0, 1, 2, 3, 4, 5, 6, 0, 0],
            [1, 2, 3, 4, 5, 6, 0, 0, 0, 0]
        ]))

    def test_compute_online_with_left_frames(self):
        step = pipeline.AddContext(left_frames=2, right_frames=0)

        # FIRST CHUNK
        data = np.array([[1, 2], [3, 4]])
        result = step.process_frames(data, 16000, offset=0, last=False)

        assert np.array_equal(result, np.array([
            [0, 0, 0, 0, 1, 2],
            [0, 0, 1, 2, 3, 4],
        ]))

        # SECOND CHUNK
        data = np.array([[5, 6], [7, 8]])
        result = step.process_frames(data, 16000, offset=2, last=True)

        assert np.array_equal(result, np.array([
            [1, 2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7, 8]
        ]))

    def test_compute_online_with_right_frames(self):
        step = pipeline.AddContext(left_frames=0, right_frames=2)

        # FIRST CHUNK
        data = np.array([[1, 2], [3, 4]])
        result = step.process_frames(data, 16000, offset=0, last=False)

        # Since it waits on context, no output is expected
        assert result is None

        # SECOND CHUNK
        data = np.array([[5, 6], [7, 8]])
        result = step.process_frames(data, 16000, offset=2, last=True)

        assert np.array_equal(result, np.array([
            [1, 2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7, 8],
            [5, 6, 7, 8, 0, 0],
            [7, 8, 0, 0, 0, 0]
        ]))

    def test_compute_online_with_left_and_right_frames(self):
        step = pipeline.AddContext(left_frames=2, right_frames=2)

        # FIRST CHUNK
        data = np.array([[1, 2], [3, 4]])
        result = step.process_frames(data, 16000, offset=0, last=False)

        # Since it waits on context, no output is expected
        assert result is None

        # SECOND CHUNK
        data = np.array([[5, 6], [7, 8]])
        result = step.process_frames(data, 16000, offset=2, last=True)

        assert np.array_equal(result, np.array([
            [0, 0, 0, 0, 1, 2, 3, 4, 5, 6],
            [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8, 0, 0],
            [3, 4, 5, 6, 7, 8, 0, 0, 0, 0]
        ]))
