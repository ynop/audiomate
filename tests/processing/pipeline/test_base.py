import unittest

import numpy as np

from audiomate.processing import pipeline


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
