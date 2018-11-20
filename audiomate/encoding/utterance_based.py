import numpy as np

from . import base


class TokenOrdinalEncoder(base.Encoder):
    """
    Class to encode labels of a given label-list. Every token of the labels is mapped to a number.
    For the full utterance a sequence/array of numbers are computed, which correspond to tokens.

    Tokens are extracted from labels by splitting using a delimiter (by default space).
    See :meth:`audiomate.annotations.Label.tokenized`.
    Hence a token can be word, phone, ..., depending on the label and the delimiter.

    Args:
        label_list_idx (str): The name of the label-list to use for encoding.
                              Only labels of this label-list are considered.
        tokens (list): List of tokens that defines the mapping. First label will get the 0 in the encoding and so on.
        token_delimiter (str): Delimiter to split labels into tokens.

    Example:

        >>> ll = LabelList(idx='words', labels=[Label('down the  road')])
        >>> utt = Utterance('utt-1', 'file-x', label_lists=ll)
        >>>
        >>> tokens = ['up', 'down', 'road', 'stree', 'the']
        >>> encoder = TokenOrdinalEncoder('words', tokens, token_delimiter=' ')
        >>> encoder.encode_utterance(utt)
        np.array([1, 4, 2])
    """

    def __init__(self, label_list_idx, tokens, token_delimiter=' '):
        self.label_list_idx = label_list_idx
        self.tokens = tokens
        self.token_delimiter = token_delimiter

    def encode_utterance(self, utterance, corpus=None):
        if self.label_list_idx not in utterance.label_lists.keys():
            raise ValueError('Utterance {} has not label-list with idx {}!'.format(utterance.idx, self.label_list_idx))

        ll = utterance.label_lists[self.label_list_idx]
        concatenated_tokens = ll.tokenized(delimiter=self.token_delimiter, overlap_threshold=0.1)

        ordinals = []

        for t in concatenated_tokens:
            if t in self.tokens:
                ordinals.append(self.tokens.index(t))
            else:
                raise ValueError('Token {} not in token-list!'.format(t))

        return np.array(ordinals)
