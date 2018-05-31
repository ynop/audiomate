import math

from . import base


class MeanVarianceNorm(base.Computation):
    """
    Pre-processing step to normalize mean and variance.

    frame = (frame - mean) / sqrt(variance)

    Args:
        mean (float): The mean to use for normalization.
        variance (float): The variance to use for normalization.s
    """

    def __init__(self, mean, variance, parent=None, name=None):
        super(MeanVarianceNorm, self).__init__(parent=parent, name=name)

        self.mean = mean
        self.variance = variance
        self.std = math.sqrt(variance)

    def compute(self, chunk, sampling_rate, corpus=None, utterance=None):
        return (chunk.data - self.mean) / self.std
