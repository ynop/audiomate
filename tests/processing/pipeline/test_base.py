import unittest

import numpy as np

from audiomate.processing import pipeline
from audiomate.processing.pipeline import base


class Multiply(pipeline.Computation):
    def __init__(self, factor, parent=None, name=None):
        super(Multiply, self).__init__(parent=parent, name=name)
        self.factor = factor

    def compute(self, data, sampling_rate, first_frame_index=0, last=False, corpus=None, utterance=None):
        return data * self.factor


class Add(pipeline.Computation):
    def __init__(self, value, parent=None, name=None):
        super(Add, self).__init__(parent=parent, name=name)
        self.value = value

    def compute(self, data, sampling_rate, first_frame_index=0, last=False, corpus=None, utterance=None):
        return data + self.value


class OfflineComputationTest(unittest.TestCase):
    def test_process(self):
        in_data = np.array([0, 1, 2, 3])
        step = Multiply(3)
        out_data = step.process_frames(in_data, 4)

        assert np.array_equal(out_data, np.array([0, 3, 6, 9]))


class Concat(pipeline.Reduction):
    def compute(self, data, sampling_rate, first_frame_index=0, last=False, corpus=None, utterance=None):
        return np.concatenate(data, axis=0)


class OfflineReductionTest(unittest.TestCase):
    def test_process(self):
        in_data = [np.array([0, 1, 2, 3]), np.array([4, 5, 6])]
        step = Concat(parents=[])
        out_data = step.process_frames(in_data, 4)

        assert np.array_equal(out_data, np.array([0, 1, 2, 3, 4, 5, 6]))


class StepTest(unittest.TestCase):
    def test_process(self):
        add_a = Add(5)
        mul = Multiply(2, parent=add_a)
        add_b = Add(2)

        concat = Concat(parents=[mul, add_b])

        in_data = np.array([0, 1, 2, 3])
        out_data = concat.process_frames(in_data, 4)

        assert np.array_equal(out_data, np.array([10, 12, 14, 16, 2, 3, 4, 5]))


class TestBuffer:

    def test_not_enough_frames_and_not_last_returns_none(self):
        buffer = base.Buffer(3, 0, 0)
        chunk = np.arange(4).reshape(2, 2)

        buffer.update(chunk, 0, False)
        res = buffer.get()

        assert res is None

    def test_not_enough_frames_and_is_last_returns_rest(self):
        buffer = base.Buffer(3, 0, 0)
        chunk = np.arange(4).reshape(2, 2)
        buffer.update(chunk, 0, True)
        res = buffer.get()

        assert np.allclose(chunk, res.data)
        assert res.offset == 0
        assert res.is_last
        assert res.left_context == 0
        assert res.right_context == 0

    def test_has_enough_returns_all(self):
        buffer = base.Buffer(3, 0, 0)
        chunk1 = np.arange(4).reshape(2, 2)
        chunk2 = np.arange(6).reshape(3, 2) + 4

        buffer.update(chunk1, 0, False)
        res = buffer.get()
        assert res is None

        buffer.update(chunk2, 2, False)
        res = buffer.get()
        assert np.allclose(np.vstack([chunk1, chunk2]), res.data)
        assert res.offset == 0
        assert not res.is_last
        assert res.left_context == 0
        assert res.right_context == 0

    def test_returns_correct_offset(self):
        buffer = base.Buffer(2, 0, 0)
        chunk1 = np.arange(4).reshape(2, 2)
        chunk2 = np.arange(6).reshape(3, 2) + 4

        buffer.update(chunk1, 0, False)
        res = buffer.get()
        assert res.offset == 0

        buffer.update(chunk2, 2, False)
        res = buffer.get()
        assert res.offset == 2

    def test_with_left_context(self):
        buffer = base.Buffer(2, 3, 0)
        chunk = np.arange(4).reshape(2, 2)

        buffer.update(chunk, 0, False)
        res = buffer.get()
        assert np.allclose(chunk, res.data)
        assert res.offset == 0
        assert not res.is_last
        assert res.left_context == 0
        assert res.right_context == 0

    def test_returns_correct_number_of_frames_with_left_context(self):
        buffer = base.Buffer(2, 3, 0)
        chunk1 = np.arange(8).reshape(4, 2)
        chunk2 = np.arange(6).reshape(3, 2) + 8

        buffer.update(chunk1, 0, False)
        res = buffer.get()
        assert np.allclose(chunk1, res.data)
        assert res.offset == 0
        assert not res.is_last
        assert res.left_context == 0
        assert res.right_context == 0

        buffer.update(chunk2, 4, False)
        res = buffer.get()
        assert np.allclose(np.vstack([chunk1[-3:], chunk2]), res.data)
        assert res.offset == 1
        assert not res.is_last
        assert res.left_context == 3
        assert res.right_context == 0

    def test_returns_correct_number_of_left_context(self):
        buffer = base.Buffer(2, 3, 0)
        chunk1 = np.arange(4).reshape(2, 2)
        chunk2 = np.arange(6).reshape(3, 2) + 4

        buffer.update(chunk1, 0, False)
        res = buffer.get()
        assert res.offset == 0
        assert not res.is_last
        assert res.left_context == 0
        assert res.right_context == 0

        buffer.update(chunk2, 2, False)
        res = buffer.get()
        assert res.offset == 0
        assert not res.is_last
        assert res.left_context == 2
        assert res.right_context == 0

    def test_dont_include_context_into_min_frames(self):
        buffer = base.Buffer(3, 1, 0)
        chunk1 = np.arange(8).reshape(4, 2)
        chunk2 = np.arange(4).reshape(2, 2) + 8
        chunk3 = np.arange(2).reshape(1, 2) + 11

        buffer.update(chunk1, 0, False)
        res = buffer.get()
        assert res.data.shape[0] == 4

        buffer.update(chunk2, 4, False)
        res = buffer.get()
        assert res is None

        buffer.update(chunk3, 6, False)
        res = buffer.get()
        assert res.data.shape[0] == 4
        assert res.offset == 3
        assert not res.is_last
        assert res.left_context == 1
        assert res.right_context == 0

    def test_not_enough_frames_and_is_last_returns_rest_with_left_context(self):
        buffer = base.Buffer(3, 2, 0)

        chunk1 = np.arange(8).reshape(4, 2)
        chunk2 = np.arange(4).reshape(2, 2) + 8

        buffer.update(chunk1, 0, False)
        res = buffer.get()
        assert np.allclose(chunk1, res.data)
        assert res.offset == 0
        assert not res.is_last
        assert res.left_context == 0
        assert res.right_context == 0

        buffer.update(chunk2, 4, True)
        res = buffer.get()
        assert np.allclose(np.vstack([chunk1[2:], chunk2]), res.data)
        assert res.offset == 2
        assert res.is_last
        assert res.left_context == 2
        assert res.right_context == 0

    def test_right_context(self):
        buffer = base.Buffer(2, 1, 2)

        chunk1 = np.arange(4).reshape(2, 2)
        chunk2 = np.arange(6).reshape(3, 2) + 4

        buffer.update(chunk1, 0, False)
        res = buffer.get()
        assert res is None

        buffer.update(chunk2, 2, False)
        res = buffer.get()
        assert np.allclose(np.vstack([chunk1, chunk2]), res.data)
        assert res.offset == 0
        assert not res.is_last
        assert res.left_context == 0
        assert res.right_context == 2

    def test_not_enough_frames_and_is_last_returns_rest_with_right_context(self):
        buffer = base.Buffer(10, 1, 2)

        chunk1 = np.arange(24).reshape(12, 2)
        chunk2 = np.arange(2).reshape(1, 2) + 24

        buffer.update(chunk1, 0, False)
        res = buffer.get()
        assert np.allclose(chunk1, res.data)
        assert res.offset == 0
        assert not res.is_last
        assert res.left_context == 0
        assert res.right_context == 2

        buffer.update(chunk2, 12, True)
        res = buffer.get()
        assert np.allclose(np.vstack([chunk1[9:], chunk2]), res.data)
        assert res.offset == 9
        assert res.is_last
        assert res.left_context == 1
        assert res.right_context == 0

    def test_multiple_buffers(self):
        buffer = base.Buffer(3, 2, 0, num_buffers=2)

        chunk1 = np.arange(8).reshape(4, 2)
        buffer.update(chunk1, 0, False, buffer_index=0)
        buffer.update(chunk1, 0, False, buffer_index=1)
        res = buffer.get()

        assert len(res.data) == 2
        assert np.allclose(chunk1, res.data[0])
        assert np.allclose(chunk1, res.data[1])
        assert res.offset == 0
        assert not res.is_last
        assert res.left_context == 0
        assert res.right_context == 0

        chunk2 = np.arange(4).reshape(2, 2) + 8
        buffer.update(chunk2, 4, True, buffer_index=0)
        res = buffer.get()

        assert res is None

        buffer.update(chunk2, 4, True, buffer_index=1)
        res = buffer.get()

        assert len(res.data) == 2
        assert np.allclose(np.vstack([chunk1[2:], chunk2]), res.data[0])
        assert np.allclose(np.vstack([chunk1[2:], chunk2]), res.data[1])
        assert res.offset == 2
        assert res.is_last
        assert res.left_context == 2
        assert res.right_context == 0

    def test_multiple_buffers_with_different_income_timings(self):
        buffer = base.Buffer(min_frames=1, left_context=0, right_context=0, num_buffers=2)

        buffer.update(np.array([[1, 2, 3, 4]]), 0, False, buffer_index=1)
        res = buffer.get()
        assert res is None

        buffer.update(np.array([[5, 6, 7, 8]]), 1, False, buffer_index=1)
        res = buffer.get()
        assert res is None

        buffer.update(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]), 0, False, buffer_index=0)
        buffer.update(np.array([[9, 10, 11, 12]]), 2, False, buffer_index=1)
        res = buffer.get()

        assert buffer.buffers[0].shape[0] == 0
        assert np.array_equal(buffer.buffers[1], np.array([[9, 10, 11, 12]]))

        assert len(res.data) == 2
        assert np.array_equal(res.data[0], np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))
        assert np.array_equal(res.data[1], np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
        assert not res.is_last
        assert res.offset == 0
        assert res.left_context == 0
        assert res.right_context == 0

        buffer.update(np.array([[8, 9, 10, 11], [12, 13, 14, 15]]), 2, True, buffer_index=0)
        buffer.update(np.array([[13, 14, 15, 16]]), 3, True, buffer_index=1)
        res = buffer.get()

        assert len(res.data) == 2
        assert np.array_equal(res.data[0], np.array([[8, 9, 10, 11], [12, 13, 14, 15]]))
        assert np.array_equal(res.data[1], np.array([[9, 10, 11, 12], [13, 14, 15, 16]]))
        assert res.is_last
        assert res.offset == 2
        assert res.left_context == 0
        assert res.right_context == 0
