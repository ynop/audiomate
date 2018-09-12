from audiomate.corpus import validation


class TestInvalidUtterancesResult:

    def test_get_report_on_pass(self):
        result = validation.InvalidUtterancesResult(passed=True, invalid_utterances={}, name='MyName', info={'a': 'b'})

        assert result.get_report() == '\n'.join([
            'MyName',
            '======',
            '',
            '--> a: b',
            '',
            'Result: Passed'
        ])

    def test_get_report_on_fail(self):
        result = validation.InvalidUtterancesResult(passed=False, invalid_utterances={
            'utt-1': 'reason1',
            'utt-4': 'reason4',
            'utt-2': 3
        }, name='MyName', info={'c': 'b', 'a': 49.0})

        assert result.get_report() == '\n'.join([
            'MyName',
            '======',
            '',
            '--> a: 49.0',
            '--> c: b',
            '',
            'Result: Failed',
            '',
            'Invalid utterances:',
            '    * utt-1 (reason1)',
            '    * utt-2 (3)',
            '    * utt-4 (reason4)'
        ])
