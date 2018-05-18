import unittest

import numpy as np

from audiomate.corpus.preprocessing.pipeline import offline


class Multiply(offline.OfflineComputation):
    def __init__(self, factor, parent=None, name=None):
        super(Multiply, self).__init__(parent=parent, name=name)
        self.factor = factor

    def compute(self, frames, sampling_rate, corpus=None, utterance=None):
        return frames * self.factor


class Add(offline.OfflineComputation):
    def __init__(self, value, parent=None, name=None):
        super(Add, self).__init__(parent=parent, name=name)
        self.value = value

    def compute(self, frames, sampling_rate, corpus=None, utterance=None):
        return frames + self.value


class OfflineComputationTest(unittest.TestCase):
    def test_process(self):
        in_data = np.array([0, 1, 2, 3])
        step = Multiply(3)
        out_data = step.process(in_data, 4)

        assert np.array_equal(out_data, np.array([0, 3, 6, 9]))


class Concat(offline.OfflineReduction):
    def compute(self, frames, sampling_rate, corpus=None, utterance=None):
        return np.concatenate(frames, axis=0)


class OfflineReductionTest(unittest.TestCase):
    def test_process(self):
        in_data = [np.array([0, 1, 2, 3]), np.array([4, 5, 6])]
        step = Concat(parents=[])
        out_data = step.process(in_data, 4)

        assert np.array_equal(out_data, np.array([0, 1, 2, 3, 4, 5, 6]))


class StepTest(unittest.TestCase):
    def test_process(self):
        add_a = Add(5)
        mul = Multiply(2, parent=add_a)
        add_b = Add(2)

        concat = Concat(parents=[mul, add_b])

        in_data = np.array([0, 1, 2, 3])
        out_data = concat.process(in_data, 4)

        assert np.array_equal(out_data, np.array([10, 12, 14, 16, 2, 3, 4, 5]))
