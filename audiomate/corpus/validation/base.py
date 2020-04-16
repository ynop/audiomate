import abc


class ValidationResult:
    """
    Representation of the result of a validation.
    The basic result just indicates a pass or fail.
    Depending on the validator it can be extended to hold more information
    (e.g. utterance-ids which triggered the task to fail).

    Args:
        passed (bool): A boolean indicating, if the validation has passed
                       (``True``) or failed (``False``).
        name (str): The name of the validator, that produced the result.
        info (dict): Dictionary containing key/value string-pairs with detailed
                     information of the validation. For example id of the
                     label-list that was validated.
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


class InvalidItemsResult(ValidationResult):
    """
    A generic result class for validators that return a list of items
    (utterances, tracks) that were classified invalid.  Besides the id of the
    item, a reason may be appended.

    Args:
        passed (bool): A boolean indicating, if the validation has passed
                       (``True``) or failed (``False``).
        invalid_items (dict): A dictionary containing item-ids, that are
                              invalid. The values are reasons why they are
                              invalid.
        name (str): The name of the validator, that produced the result.
        info (dict): Dictionary containing key/value string-pairs with detailed
                     information of the validation. For example id of the
                     label-list that was validated.
    """

    def __init__(self, passed, invalid_items, name='Validation', item_name='Utterances', info=None):
        super(InvalidItemsResult, self).__init__(passed, name=name, info=info)

        self.invalid_items = invalid_items
        self.item_name = item_name

    def get_report(self):
        """
        Return a string containing a report of the result.
        This can used to print or save to a text file.

        Returns:
            str: String containing infos about the result
        """

        lines = [super(InvalidItemsResult, self).get_report()]

        if len(self.invalid_items) > 0:
            lines.append('\nInvalid {}:'.format(self.item_name))

            sorted_items = sorted(self.invalid_items.items(), key=lambda x: x[0])
            lines.extend(['    * {} ({})'.format(x, y) for x, y in sorted_items])

        return '\n'.join(lines)


class Validator(abc.ABC):
    """
    A validator is a class that tests a specific behaviour/state
    of a corpus.
    """

    @abc.abstractmethod
    def name(self):
        """ Return a name, identifying the task. """
        raise NotImplementedError()

    @abc.abstractmethod
    def validate(self, corpus_to_validate):
        """
        Perform the validation on the given corpus.

        Args:
            corpus (Corpus): The corpus to test/validate.

        Returns:
            ValidationResult: The result containing at least the
                              pass/fail indication.
        """
        raise NotImplementedError()
