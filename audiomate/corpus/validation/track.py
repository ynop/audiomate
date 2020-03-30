import multiprocessing

import audioread

from . import base
from audiomate import logutil

logger = logutil.getLogger()


class TrackReadValidator(base.Validator):
    """
    Check if the track can be opened and read.
    By reading the first few samples.
    """

    def __init__(self, num_workers=1):
        self.num_workers = num_workers

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

        with multiprocessing.pool.ThreadPool(self.num_workers) as p:
            result = list(logger.progress(
                p.imap(
                    self.validate_track,
                    list(corpus.tracks.values())
                ),
                total=corpus.num_tracks,
                description='Validate tracks'
            ))

        invalid_tracks = {x[0]: x[1] for x in result if x[1] is not None}
        passed = len(invalid_tracks) <= 0

        return base.InvalidItemsResult(
            passed,
            invalid_tracks,
            item_name='Tracks',
            name=self.name()
        )

    def validate_track(self, track):
        result = None

        try:
            track.read_samples(duration=0.001)
        except EOFError:
            result = 'EOFError'
        except audioread.NoBackendError:
            result = 'NoBackendError'
        except Exception as ex:
            result = str(ex)

        return (track.idx, result)
