import copy
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
                     (default: inf) (inf defines the end of the utterance)
        meta (dict): A dictionary containing additional information
                     for the label.

    Attributes:
        label_list (LabelList): The label-list this label is belonging to.
    """
    __slots__ = ['value', 'start', 'end', 'label_list', 'meta']

    def __init__(self, value, start=0, end=float('inf'), meta=None):
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
        data_this = (self.start, self.end, self.value.lower())
        data_other = (other.start, other.end, other.value.lower())

        return data_this < data_other

    def __repr__(self) -> str:
        return 'Label({}, {}, {})'.format(self.value, self.start, self.end)

    def __copy__(self):
        return Label(
            self.value,
            start=self.start,
            end=self.end,
            meta=self.meta
        )

    def __deepcopy__(self, memo):
        return Label(
            self.value,
            start=self.start,
            end=self.end,
            meta=copy.deepcopy(self.meta, memo)
        )

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
        elif self.end == float('inf'):
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

        if self.end != float('inf') or self.label_list.utterance.end >= 0:
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

    def do_overlap(self, other_label, adjacent=True):
        """
        Determine whether ``other_label`` overlaps with this label.
        If ``adjacent==True``, adjacent labels are also considered as overlapping.

        Args:
            other_label (Label): Another label.
            adjacent (bool): If ``True``, adjacent labels are
                             considered as overlapping.

        Returns:
            bool: ``True`` if the two labels overlap, ``False`` otherwise.
        """
        this_end = self.end
        other_end = other_label.end

        if this_end == float('inf'):
            this_end = self.end_abs

        if other_end == float('inf'):
            other_end = other_label.end_abs

        if adjacent:
            first_ends_before = this_end < other_label.start
            second_ends_before = other_end < self.start
        else:
            first_ends_before = this_end <= other_label.start
            second_ends_before = other_end <= self.start

        return not (first_ends_before or second_ends_before)

    def overlap_duration(self, other_label):
        """
        Return the duration of the overlapping part between this label
        and ``other_label``.

        Args:
            other_label(Label): Another label to check.

        Return:
            float: The duration of overlap in seconds.

        Example:
            >>> label_a = Label('a', 3.4, 5.6)
            >>> label_b = Label('b', 4.8, 6.2)
            >>> label_a.overlap_duration(label_b)
            0.8
        """
        this_end = self.end
        other_end = other_label.end

        if this_end == float('inf'):
            this_end = self.end_abs

        if other_end == float('inf'):
            other_end = other_label.end_abs

        start_overlap = max(self.start, other_label.start)
        end_overlap = min(this_end, other_end)

        return max(0, end_overlap - start_overlap)
