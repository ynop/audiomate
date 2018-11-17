from functools import total_ordering


@total_ordering
class Label(object):
    """
    Represents a label that describes some part of an utterance.

    Args:
        value (str): The text of the label.
        start (float): Start of the label within the utterance in seconds.
                       (default: 0)
        end (float): End of the label within the utterance in seconds.
                     (default: -1) (-1 defines the end of the utterance)
        meta (dict): A dictionary containing additional information
                     for the label.

    Attributes:
        label_list (LabelList): The label-list this label is belonging to.
    """
    __slots__ = ['value', 'start', 'end', 'label_list', 'meta']

    def __init__(self, value, start=0, end=-1, meta=None):
        self.value = value
        self.start = start
        self.end = end
        self.meta = meta or {}
        self.label_list = None

    def __eq__(self, other):
        data_this = (self.start, self.end, self.value.lower())
        data_other = (other.start, other.end, other.value.lower())
        return data_this == data_other

    def __lt__(self, other):
        self_end = float('inf') if self.end == -1 else self.end
        other_end = float('inf') if other.end == -1 else other.end

        data_this = (self.start, self_end, self.value.lower())
        data_other = (other.start, other_end, other.value.lower())

        return data_this < data_other

    def __repr__(self) -> str:
        return 'Label({}, {}, {})'.format(self.value, self.start, self.end)

    @property
    def start_abs(self):
        """
        Return the absolute start of the label in seconds relative to
        the signal. If the label isn't linked to any utterance via label-list,
        it is assumed ``self.start`` is relative to the start of the signal,
        hence ``self.start`` == ``self.start_abs``.
        """
        if self.label_list is None or self.label_list.utterance is None:
            return self.start

        return self.label_list.utterance.start + self.start

    @property
    def end_abs(self):
        """
        Return the absolute end of the label in seconds relative to the signal.
        If the label isn't linked to any utterance via label-list,
        it is assumed ``self.end`` is relative to the start of the signal,
        hence ``self.end`` == ``self.end_abs``.
        """
        if self.label_list is None or self.label_list.utterance is None:
            return self.end
        elif self.end == -1:
            return self.label_list.utterance.end_abs
        else:
            return self.end + self.label_list.utterance.start

    @property
    def duration(self):
        """
        Return the duration of the label in seconds.
        """
        return self.end_abs - self.start_abs

    def read_samples(self, sr=None):
        """
        Read the samples of the utterance.

        Args:
            sr (int): If None uses the sampling rate given by the track,
                      otherwise resamples to the given sampling rate.

        Returns:
            np.ndarray: A numpy array containing the samples as a
            floating point (numpy.float32) time series.
        """
        duration = None

        if self.end >= 0 or self.label_list.utterance.end >= 0:
            duration = self.duration

        track = self.label_list.utterance.track

        return track.read_samples(
            sr=sr,
            offset=self.start_abs,
            duration=duration
        )

    def tokenized(self, delimiter=' '):
        """
        Return a list with tokens from the value of the label.
        Tokens are extracted by splitting the string using ``delimiter`` and
        then trimming any whitespace before and after splitted strings.

        Args:
            delimiter (str): The delimiter used to split into tokens.
                             (default: space)

        Return:
            list: A list of tokens in the order they occur in the label.

        Examples:

            >>> label = Label('as is oh')
            >>> label.tokenized()
            ['as', 'is', 'oh']

        Using a different delimiter (whitespace is trimmed anyway):

            >>> label = Label('oh hi, as, is  ')
            >>> label.tokenized(delimiter=',')
            ['oh hi', 'as', 'is']
        """

        tokens = self.value.split(sep=delimiter)
        tokens = [t.strip() for t in tokens]

        while '' in tokens:
            tokens.remove('')

        return tokens
