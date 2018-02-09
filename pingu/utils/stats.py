import numpy as np


class DataStats(object):
    """
    This class holds statistics for any kind of numerical data.

    Args:
        mean (float): The mean overall data points.
        var (float): The variance overall data points.
        min (float): The minimum value within all data points.
        max (float): The maximum value within all data points.
        num (int): The number of data points, these statistics were calculated of.
    """

    __slots__ = ['mean', 'var', 'min', 'max', 'num']

    def __init__(self, mean, var, min, max, num):
        self.mean = mean
        self.var = var
        self.min = min
        self.max = max
        self.num = num

    @property
    def values(self):
        """
        Return all values as numpy-array (mean, var, min, max, num).
        """
        return np.array([self.mean, self.var, self.min, self.max, self.num])

    def to_dict(self):
        """
        Return the stats as a dictionary.
        """
        return {
            'mean': self.mean,
            'var': self.var,
            'min': self.min,
            'max': self.max,
            'num': self.num
        }

    @classmethod
    def from_dict(self, dict_with_stats):
        """
        Create a DataStats object from a dictionary with stats.

        Args:
            dict_with_stats (dict): Dictionary containing stats.

        Returns:
            (DataStats): Statistics
        """

        return DataStats(dict_with_stats['mean'],
                         dict_with_stats['var'],
                         dict_with_stats['min'],
                         dict_with_stats['max'],
                         dict_with_stats['num'])

    @classmethod
    def concatenate(cls, list_of_stats):
        """
        Take a list of stats from different sets of data points and
        merge the stats for getting stats overall data points.

        Args:
            list_of_stats (iterable): A list containing stats for different sets of data points.

        Returns:
            DataStats: Stats calculated overall sets of data points.
        """

        all_stats = np.stack([stats.values for stats in list_of_stats])
        all_counts = all_stats[:, 4]
        all_counts_relative = all_counts / np.sum(all_counts)

        min_value = float(np.min(all_stats[:, 2]))
        max_value = float(np.max(all_stats[:, 3]))
        mean_value = float(np.sum(all_counts_relative * all_stats[:, 0]))
        var_value = float(np.sum(all_counts_relative * (all_stats[:, 1] + np.power(all_stats[:, 0] - mean_value, 2))))
        num_value = int(np.sum(all_counts))

        return cls(mean_value, var_value, min_value, max_value, num_value)
