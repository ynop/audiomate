from . import base


class CombinedValidationResult(base.ValidationResult):
    """
    Result of running multiple validation-tasks with the validator.

    Args:
        passed (bool): A boolean, indicating if all tasks have passed (True) or at least one failed (False).
        results (dict): A dictionary containing the results of all validators, with the task name as key.
        info (dict): Dictionary containing key/value string-pairs with detailed information of the validation.
                     For example id of the label-list that was validated.
    """

    def __init__(self, passed, results=None, info=None):
        super(CombinedValidationResult, self).__init__(passed, info=info)

        self.results = results or {}

    def get_report(self):
        sorted_val = sorted(self.results.items(), key=lambda x: x[0])
        lines = ['{} --> {}'.format(x, 'Passed' if y.passed else 'Failed') for x, y in sorted_val]
        lines.append('\n')

        for name in sorted(self.results.keys()):
            result = self.results[name]

            lines.append(result.get_report())
            lines.append('\n')

        return '\n'.join(lines)


class CombinedValidator(base.Validator):
    """
    The CombinedValidator is used to execute multiple validators at once.

    Args:
        validators (list): A list of validators that are executed.
    """

    def __init__(self, validators=None):
        self.validators = validators or []

    def name(cls):
        return 'Combined-Validator'

    def validate(self, corpus):
        """
        Perform validation on the given corpus.

        Args:
            corpus (Corpus): The corpus to test/validate.
        """

        passed = True
        results = {}

        for validator in self.validators:
            sub_result = validator.validate(corpus)
            results[validator.name()] = sub_result

            if not sub_result.passed:
                passed = False

        return CombinedValidationResult(passed, results)
