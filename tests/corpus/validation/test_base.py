from audiomate.corpus import validation


class TestInvalidItemsResult:

    def test_get_report_on_pass(self):
        result = validation.InvalidItemsResult(passed=True, invalid_items={}, name='MyName', info={'a': 'b'})

        assert result.get_report() == '\n'.join([
            'MyName',
            '======',
            '',
            '--> a: b',
            '',
            'Result: Passed'
        ])

    def test_get_report_on_fail(self):
        result = validation.InvalidItemsResult(passed=False, invalid_items={
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
            'Invalid Utterances:',
            '    * utt-1 (reason1)',
            '    * utt-2 (3)',
            '    * utt-4 (reason4)'
        ])

    def test_get_report_on_fail_with_type_name(self):
        result = validation.InvalidItemsResult(
            passed=False,
            invalid_items={
                'utt-1': 'reason1',
                'utt-4': 'reason4',
                'utt-2': 3
            },
            name='MyName',
            item_name='Tracks',
            info={'c': 'b', 'a': 49.0}
        )

        assert result.get_report() == '\n'.join([
            'MyName',
            '======',
            '',
            '--> a: 49.0',
            '--> c: b',
            '',
            'Result: Failed',
            '',
            'Invalid Tracks:',
            '    * utt-1 (reason1)',
            '    * utt-2 (3)',
            '    * utt-4 (reason4)'
        ])
