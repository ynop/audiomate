import librosa
import audioread
import numpy as np


def process_buffer(buffer, n_channels, src_sr, target_sr):
    """
    Merge the read blocks and resample if necessary.

    Args:
        buffer (list): A list of blocks of samples.
        n_channels (int): The number of channels of the input data.
        src_sr (int): The sampling-rate of the input data.
        target_sr (int): The desired sampling-rate to return the data.

    Returns:
        (np.array, int): The samples and the sampling-rate.
    """
    samples = np.concatenate(buffer)

    if n_channels > 1:
        samples = samples.reshape((-1, n_channels)).T
        samples = librosa.to_mono(samples)

    if target_sr is not None and src_sr != target_sr:
        samples = librosa.resample(samples, src_sr, target_sr)

    return samples, target_sr


def read_blocks(file_path, sr_target=None, start=0.0, end=-1.0, buffer_size=5760000):
    """
    Read an audio file block after block. The blocks are yielded one by one.

    Args:
        file_path (str): Path to the file to read.
        sr_target (int): The sampling-rate to resample the audio to. None uses the native sampling-rate.
        start (float): Start in seconds to read from.
        end (float): End in seconds to read to. -1.0 means to the end of the file.
        buffer_size (int): Number of samples to load into memory at once and return as a single block.
                           The exact number of loaded samples depends on the block-size of the audioread library.
                           So it can be of x higher, where the x is typically 1024 or 4096.

    Returns:
        Generator: A generator yielding a tuple for every block. First item are the actual samples.
                   The second item is the sampling-rate of the samples.
    """
    buffer = []
    n_buffer = 0
    n_samples = 0

    with audioread.audio_open(file_path) as input_file:
        n_channels = input_file.channels
        sr_native = input_file.samplerate
        sr_target = sr_target or sr_native

        start_sample = int(np.round(sr_native * start)) * n_channels

        if end > 0:
            end_sample = int(np.round(sr_native * end)) * n_channels
        else:
            end_sample = np.inf

        for block in input_file:
            block = librosa.util.buf_to_float(block)
            n_prev = n_samples
            n_samples += len(block)

            if n_samples < start_sample:
                continue

            if n_prev > end_sample:
                break

            if n_samples > end_sample:
                block = block[:end_sample - n_prev]

            if n_prev <= start_sample <= n_samples:
                block = block[start_sample - n_prev:]

            n_buffer += len(block)
            buffer.append(block)

            if n_buffer >= buffer_size:
                yield process_buffer(buffer, n_channels, sr_native, sr_target)

                buffer = []
                n_buffer = 0

        if len(buffer) > 0:
            yield process_buffer(buffer, n_channels, sr_native, sr_target)


def read_frames(file_path, frame_size, hop_size, sr_target=None, start=0.0, end=-1.0, buffer_size=5760000):
    """
    Read an audio file frame by frame. The frames are yielded one after another.

    Args:
        file_path (str): Path to the file to read.
        frame_size (int): The number of samples per frame.
        hop_size (int): The number of samples between two frames.
        sr_target (int): The sampling-rate to resample the audio to. None uses the native sampling-rate.
        start (float): Start in seconds to read from.
        end (float): End in seconds to read to. -1.0 means to the end of the file.
        buffer_size (int): Number of samples to load into memory at once and return as a single block.
                           The exact number of loaded samples depends on the block-size of the audioread library.
                           So it can be of x higher, where the x is typically 1024 or 4096.

    Returns:
        Generator: A generator yielding a tuple for every frame. The first item is the frame,
                   the second the sampling-rate and the third a boolean indicating if it is the last frame.
    """
    rest_samples = np.array([], dtype=np.float32)

    for block, output_sr in read_blocks(file_path, sr_target=sr_target, start=start, end=end, buffer_size=buffer_size):

        # Prepend rest samples from previous block
        block = np.concatenate([rest_samples, block])

        current_sample = 0

        # Get frames that are fully contained in the block
        while current_sample + frame_size < block.size:
            frame = block[current_sample:current_sample + frame_size]
            yield frame, output_sr, False
            current_sample += hop_size

        # Store rest samples for next block
        rest_samples = block[current_sample:]

    if rest_samples.size > 0:
        rest_samples = np.pad(rest_samples, (0, frame_size - rest_samples.size), mode='constant', constant_values=0)
        yield rest_samples, output_sr, True
