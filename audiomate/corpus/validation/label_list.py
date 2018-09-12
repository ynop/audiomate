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
