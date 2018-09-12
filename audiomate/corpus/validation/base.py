import abc


class ValidationResult:
    """
    Representation of the result of a validation.
    The basic result just indicates a pass or fail.
    Depending on the validator it can be extended to hold more information
    (e.g. utterance-ids which triggered the task to fail).

    Args:
        passed (bool): A boolean indicating, if the validation has passed (True) or failed (False).
        name (str): The name of the validator, that produced the result.
        info (dict): Dictionary containing key/value string-pairs with detailed information of the validation.
                     For example id of the label-list that was validated.
    """

    def __init__(self, passed, name='Validation', info=None):
        self.passed = passed
        self.name = name
        self.info = info

    def get_report(self):
        """
        Return a string containing a report of the result.
        This can used to print or save to a text file.

        Returns:
            str: String containing infos about the result
        """

        lines = [
            self.name,
            '=' * len(self.name)
        ]

        if self.info is not None:
            lines.append('')
            sorted_info = sorted(self.info.items(), key=lambda x: x[0])
            lines.extend(['--> {}: {}'.format(k, v) for k, v in sorted_info])

        lines.append('')
        lines.append('Result: {}'.format('Passed' if self.passed else 'Failed'))

        return '\n'.join(lines)


class InvalidUtterancesResult(ValidationResult):
    """
    A generic result class for validators that return a list of utterances that were classified invalid.
    Besides the utterance-id, a reason may be appended.

    Args:
        passed (bool): A boolean indicating, if the validation has passed (True) or failed (False).
        invalid_utterances (dict): A dictionary containing utterance-ids, that are invalid.
                                   The values are reasons why they are invalid.
        name (str): The name of the validator, that produced the result.
        info (dict): Dictionary containing key/value string-pairs with detailed information of the validation.
                     For example id of the label-list that was validated.
    """

    def __init__(self, passed, invalid_utterances, name='Validation', info=None):
        super(InvalidUtterancesResult, self).__init__(passed, name=name, info=info)

        self.invalid_utterances = invalid_utterances

    def get_report(self):
        """
        Return a string containing a report of the result.
        This can used to print or save to a text file.

        Returns:
            str: String containing infos about the result
        """

        lines = [super(InvalidUtterancesResult, self).get_report()]

        if len(self.invalid_utterances) > 0:
            lines.append('\nInvalid utterances:')

            sorted_items = sorted(self.invalid_utterances.items(), key=lambda x: x[0])
            lines.extend(['    * {} ({})'.format(x, y) for x, y in sorted_items])

        return '\n'.join(lines)


class Validator(abc.ABC):
    """
    A validator is a class that tests a specific behaviour/state of a corpus.
    """

    @abc.abstractmethod
    def name(self):
        """ Return a name, identifying the task. """
        pass

    @abc.abstractmethod
    def validate(self, corpus):
        """
        Perform the validation on the given corpus.

        Args:
            corpus (Corpus): The corpus to test/validate.

        Returns:
            ValidationResult: The result containing at least the pass/fail indication.
        """
        pass
