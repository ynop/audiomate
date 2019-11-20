import functools
import multiprocessing

import sox
import soundfile as sf
from tqdm import tqdm

from . import base


class WavAudioFileConverter(base.AudioFileConverter):
    """
    Class that creates a new instance of a corpus,
    so that all audio files meet given requirements.
    """

    def __init__(self, num_workers=4, sampling_rate=16000, separate_file_per_utterance=False,
                 force_conversion=False):
        super(WavAudioFileConverter, self).__init__(
            sampling_rate,
            separate_file_per_utterance,
            force_conversion
        )

        self.num_workers = num_workers

        self.expected_properties = {
            'samplerate': self.sampling_rate,
            'format': 'WAV',
            'subtype': 'PCM_16'
        }

    def _file_extension(self):
        return 'wav'

    def _does_utt_match_target_format(self, utterance):
        """
        Return ``True`` if the utterance already matches the target format,
        ``False`` otherwise.
        """
        if utterance.track.path.endswith('mp3'):
            return False

        try:
            info = sf.info(utterance.track.path)

            for key, value in self.expected_properties.items():
                if info.__getattribute__(key) != value:
                    return False

        except RuntimeError:
            return False

        return True

    def _convert_files(self, files):
        """
        Store the given samples with the target format
        at ``path``.
        """
        print(files)
        with multiprocessing.Pool(self.num_workers) as p:
            func = functools.partial(
                _process_file,
                target_sr=self.sampling_rate
            )
            list(tqdm(p.imap(func, list(files)), total=len(files)))


def _process_file(file_item, target_sr):
    src = file_item[0]
    start = file_item[1]
    end = file_item[2]
    target = file_item[3]

    tfm = sox.Transformer()

    if start > 0 and end == float('inf'):
        tfm.trim(start)
    elif end != float('inf'):
        tfm.trim(start, end)

    tfm.convert(target_sr, 1, 16)
    tfm.build(src, target)
