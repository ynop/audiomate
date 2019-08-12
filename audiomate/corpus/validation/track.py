from . import base

import audioread


class TrackReadValidator(base.Validator):
    """
    Check if the track can be opened and read.
    """

    def name(self):
        return 'Track-Read'

    def validate(self, corpus):
        """
        Perform the validation on the given corpus.

        Args:
            corpus (Corpus): The corpus to test/validate.

        Returns:
            InvalidItemsResult: Validation result.
        """

        invalid_tracks = {}

        for track in corpus.tracks.values():
            try:
                track.duration
            except EOFError:
                invalid_tracks[track.idx] = 'EOFError'
            except audioread.NoBackendError:
                invalid_tracks[track.idx] = 'NoBackendError'

        passed = len(invalid_tracks) <= 0
        return base.InvalidItemsResult(
            passed,
            invalid_tracks,
            item_name='Tracks',
            name=self.name()
        )
