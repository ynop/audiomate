from audiomate.corpus import validation

from tests import resources


def create_mock_validator(name, passed):
    class MockValidator(validation.Validator):

        def __init__(self, passed):
            self.passed = passed
            self.called = None

        def name(self):
            return name

        def validate(self, corpus):
            self.called = corpus
            return validation.ValidationResult(self.passed, name=self.name())

    return MockValidator(passed)


class TestCombinedValidator:

    def test_validate_does_not_pass(self):
        cv = validation.CombinedValidator(validators=[
            create_mock_validator('a', True),
            create_mock_validator('b', True),
            create_mock_validator('c', False),
            create_mock_validator('d', True)
        ])

        test_corpus = resources.create_dataset()
        result = cv.validate(test_corpus)

        assert not result.passed
        assert len(result.results) == 4

        assert result.results['a'].passed
        assert result.results['b'].passed
        assert not result.results['c'].passed
        assert result.results['d'].passed

        for i in range(4):
            assert cv.validators[i].called == test_corpus


class TestCombinedValidationResult:

    def test_get_report(self):
        result = validation.CombinedValidationResult(passed=False,
                                                     results={
                                                         'a': validation.ValidationResult(passed=True, name='a'),
                                                         'c': validation.ValidationResult(passed=False, name='c'),
                                                         'b': validation.ValidationResult(passed=True, name='b'),
                                                         'd': validation.ValidationResult(passed=True, name='d')
                                                     })

        assert result.get_report() == '\n'.join([
            'a --> Passed',
            'b --> Passed',
            'c --> Failed',
            'd --> Passed',
            '',
            '',
            'a',
            '=',
            '',
            'Result: Passed',
            '',
            '',
            'b',
            '=',
            '',
            'Result: Passed',
            '',
            '',
            'c',
            '=',
            '',
            'Result: Failed',
            '',
            '',
            'd',
            '=',
            '',
            'Result: Passed',
            '',
            ''
        ])
