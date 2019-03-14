from audiomate import corpus
from . import base


class UtteranceTranscriptionRatioValidator(base.Validator):
    """
    Checks if the ratio between utterance-duration and transcription-length is below a given ratio.
    This is used to find utterances where the speech transcription is to long for a given utterance,
    meaning too much characters per second.

    Args:
        max_characters_per_second (int): If char/sec of an utterance is higher than this it is returned.
        label_list_idx (str): The label-list to use for validation.
    """

    def __init__(self, max_characters_per_second=10, label_list_idx=corpus.LL_WORD_TRANSCRIPT):
        self.max_characters_per_second = max_characters_per_second
        self.label_list_idx = label_list_idx

    def name(self):
        return 'Utterance-Transcription-Ratio ({})'.format(self.label_list_idx)

    def validate(self, corpus):
        """
        Perform the validation on the given corpus.

        Args:
            corpus (Corpus): The corpus to test/validate.

        Returns:
            InvalidUtterancesResult: Validation result.
        """
        invalid_utterances = {}

        for utterance in corpus.utterances.values():
            duration = utterance.duration
            ll = utterance.label_lists[self.label_list_idx]

            # We count the characters of all labels
            transcription = ' '.join([l.value for l in ll])
            num_chars = len(transcription.replace(' ', ''))

            char_per_sec = num_chars / duration

            if char_per_sec > self.max_characters_per_second:
                invalid_utterances[utterance.idx] = char_per_sec

        passed = len(invalid_utterances) <= 0
        info = {
            'Threshold max. characters per second': str(self.max_characters_per_second),
            'Label-List ID': self.label_list_idx
        }

        return base.InvalidUtterancesResult(passed, invalid_utterances, name=self.name(), info=info)


class LabelCountValidator(base.Validator):
    """
    Checks if every utterance contains a label-list with the given id and has at least `min_number_of_labels`.

    Args:
        min_number_of_labels (int): Minimum number of expected labels.
        label_list_idx (str): The label-list to use for validation.
    """

    def __init__(self, min_number_of_labels=1, label_list_idx=corpus.LL_WORD_TRANSCRIPT):
        self.min_number_of_labels = min_number_of_labels
        self.label_list_idx = label_list_idx

    def name(self):
        return 'Label-Count ({})'.format(self.label_list_idx)

    def validate(self, corpus):
        """
        Perform the validation on the given corpus.

        Args:
            corpus (Corpus): The corpus to test/validate.

        Returns:
            InvalidUtterancesResult: Validation result.
        """
        invalid_utterances = {}

        for utterance in corpus.utterances.values():
            if self.label_list_idx in utterance.label_lists.keys():
                ll = utterance.label_lists[self.label_list_idx]

                if len(ll) < self.min_number_of_labels:
                    invalid_utterances[utterance.idx] = 'Only {} labels'.format(len(ll))
            else:
                invalid_utterances[utterance.idx] = 'No label-list {}'.format(self.label_list_idx)

        passed = len(invalid_utterances) <= 0
        info = {
            'Min. number of labels': str(self.min_number_of_labels),
            'Label-List ID': self.label_list_idx
        }

        return base.InvalidUtterancesResult(passed, invalid_utterances, name=self.name(), info=info)


class LabelCoverageValidationResult(base.ValidationResult):
    """
    Result of a the :class:`LabelCoverageValidator`.

    Args:
        passed (bool): A boolean indicating, if the validation has passed (True) or failed (False).
        uncovered_segments (dict): A dictionary containing a list of uncovered segments for every utterance.
        name (str): The name of the validator, that produced the result.
        info (dict): Dictionary containing key/value string-pairs with detailed information of the validation.
                     For example id of the label-list that was validated.
    """

    def __init__(self, passed, uncovered_segments, name, info=None):
        super(LabelCoverageValidationResult, self).__init__(passed, name=name, info=info)

        self.uncovered_segments = uncovered_segments

    def get_report(self):
        """
        Return a string containing a report of the result.
        This can used to print or save to a text file.

        Returns:
            str: String containing infos about the result
        """

        lines = [super(LabelCoverageValidationResult, self).get_report()]

        if len(self.uncovered_segments) > 0:
            lines.append('\nUncovered segments:')

            for utt_idx, utt_segments in self.uncovered_segments.items():
                if len(utt_segments) > 0:
                    lines.append('\n{}'.format(utt_idx))
                    sorted_items = sorted(utt_segments, key=lambda x: x[0])
                    lines.extend(['    * {:10.2f}  -  {:10.2f}'.format(x[0], x[1]) for x in sorted_items])

        return '\n'.join(lines)


class LabelCoverageValidator(base.Validator):
    """
    Check if every portion of the utterance is covered with at least one label.
    The validator returns segments (start, end) of an utterance, where no label is defined
    within the given label-list.

    Args:
        label_list_idx (str): The idx of the label-list to check.
        threshold (float): A threshold for the length of a segment to be considered as uncovered.
    """

    def __init__(self, label_list_idx, threshold=0.01):
        self.label_list_idx = label_list_idx
        self.threshold = threshold

    def name(self):
        return 'Label-Coverage ({})'.format(self.label_list_idx)

    def validate(self, corpus):
        """
        Perform the validation on the given corpus.

        Args:
            corpus (Corpus): The corpus to test/validate.

        Returns:
            InvalidUtterancesResult: Validation result.
        """

        uncovered_segments = {}

        for utterance in corpus.utterances.values():
            utt_segments = self.validate_utterance(utterance)

            if len(utt_segments) > 0:
                uncovered_segments[utterance.idx] = utt_segments

        passed = len(uncovered_segments) <= 0
        info = {
            'Label-List ID': self.label_list_idx,
            'Threshold': str(self.threshold)
        }

        return LabelCoverageValidationResult(passed, uncovered_segments, self.name(), info)

    def validate_utterance(self, utterance):
        """
        Validate the given utterance and return a list of uncovered segments (start, end).
        """
        uncovered_segments = []

        if self.label_list_idx in utterance.label_lists.keys():
            start = 0
            end = utterance.duration
            ll = utterance.label_lists[self.label_list_idx]
            ranges = list(ll.ranges(yield_ranges_without_labels=True))

            # Check coverage at start
            if ranges[0][0] - start > self.threshold:
                uncovered_segments.append((start, ranges[0][0]))

            # Check for empty ranges
            for range in ranges:
                if len(range[2]) == 0 and range[1] - range[0] > self.threshold:
                    uncovered_segments.append((range[0], range[1]))

            # Check coverage at end
            if ranges[-1][1] > 0 and end - ranges[-1][1] > self.threshold:
                uncovered_segments.append((ranges[-1][1], end))

        else:
            uncovered_segments.append((utterance.start, utterance.end))

        return uncovered_segments


class LabelOverflowValidationResult(base.ValidationResult):
    """
    Result of a the :class:`LabelOverflowValidator`.

    Args:
        passed (bool): A boolean indicating, if the validation has passed (True) or failed (False).
        overflow_segments (dict): A dictionary containing a list of overflowing segments for every utterance.
        name (str): The name of the validator, that produced the result.
        info (dict): Dictionary containing key/value string-pairs with detailed information of the validation.
                     For example id of the label-list that was validated.
    """

    def __init__(self, passed, overflow_segments, name, info=None):
        super(LabelOverflowValidationResult, self).__init__(passed, name=name, info=info)

        self.overflow_segments = overflow_segments

    def get_report(self):
        """
        Return a string containing a report of the result.
        This can used to print or save to a text file.

        Returns:
            str: String containing infos about the result
        """

        lines = [super(LabelOverflowValidationResult, self).get_report()]

        if len(self.overflow_segments) > 0:
            lines.append('\nSegments outside of the utterance:')

            for utt_idx, utt_segments in self.overflow_segments.items():
                if len(utt_segments) > 0:
                    lines.append('\n{}'.format(utt_idx))
                    sorted_items = sorted(utt_segments, key=lambda x: x[0])
                    lines.extend(['    * {:10.2f}  -  {:10.2f} :  {}'.format(x[0], x[1], x[2]) for x in sorted_items])

        return '\n'.join(lines)


class LabelOverflowValidator(base.Validator):
    """
    Check if all labels are within the boundaries of an utterance.
    Finds all segments of labels that lie outside of an utterance.

    Args:
        label_list_idx (str): The idx of the label-list to check.
        threshold (float): A threshold for a time distance to be considered for an overflow.
    """

    def __init__(self, label_list_idx, threshold=0.01):
        self.label_list_idx = label_list_idx
        self.threshold = threshold

    def name(self):
        return 'Label-Overflow ({})'.format(self.label_list_idx)

    def validate(self, corpus):
        """
        Perform the validation on the given corpus.

        Args:
            corpus (Corpus): The corpus to test/validate.

        Returns:
            InvalidUtterancesResult: Validation result.
        """

        overflow_segments = {}

        for utterance in corpus.utterances.values():
            utt_segments = self.validate_utterance(utterance)

            if len(utt_segments) > 0:
                overflow_segments[utterance.idx] = utt_segments

        passed = len(overflow_segments) <= 0
        info = {
            'Label-List ID': self.label_list_idx,
            'Threshold': str(self.threshold)
        }

        return LabelOverflowValidationResult(passed, overflow_segments, self.name(), info)

    def validate_utterance(self, utterance):
        """
        Validate the given utterance and return a list of segments (start, end, label-value),
        that are outside of the utterance.
        """
        overflow_segments = []

        if self.label_list_idx in utterance.label_lists.keys():
            ll = utterance.label_lists[self.label_list_idx]
            start = 0
            end = utterance.duration

            for label in ll:
                if start - label.start > self.threshold:
                    label_end = label.end if label.end != float('inf') else end
                    overflow_end = min(start, label_end)
                    overflow_segments.append((label.start, overflow_end, label.value))

                if label.end != float('inf') and label.end - end > self.threshold:
                    overflow_start = max(end, label.start)
                    overflow_segments.append((overflow_start, label.end, label.value))

        return overflow_segments
