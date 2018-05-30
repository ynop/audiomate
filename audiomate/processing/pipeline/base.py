import abc

import numpy as np
import networkx as nx

from audiomate import processing


class Chunk(object):
    """
    Represents a chunk of data. It is used to pass data between different steps of a pipeline.

    Args:
        data (np.ndarray or list): A single array of frames or a list of separate chunks of frames of equal size.
        offset (int): The index of the first frame in the chunk within the sequence.
        is_last (bool): Whether this is the last chunk of the sequence.
        left_context (int): Number of frames that act as context at the begin of the chunk (left).
        right_context (int): Number of frames that act as context at the end of the chunk (right).
    """

    def __init__(self, data, offset, is_last, left_context=0, right_context=0):
        self.data = data
        self.offset = offset
        self.is_last = is_last
        self.left_context = left_context
        self.right_context = right_context

    def __repr__(self):
        return 'Chunk(data [{}], offset [{}], is-last [{}], left[{}], right[{}])'.format(self.data.shape,
                                                                                         self.offset,
                                                                                         self.is_last,
                                                                                         self.left_context,
                                                                                         self.right_context)


class Buffer(object):
    """
    The buffer is a utility to store frames if there are not enough frames as required by some pipeline step.
    The incoming frames are added to the buffer using the ``update`` method.
    The buffer then determines if there are enough frames for a new chunk. If so a chunk is returned,
    otherwise the data is appended to the buffer.

    Context at the beginning or end of the sequence is not padded and is indicated by the chunks context attributes,
    which for example will be 0 for the left context in the first chunk.

    Using ``num_buffers`` multiple parallel buffers can be used. This means that a chunk is only returned,
    if all buffers have enough frames.

    Args:
        min_frames (int): The minimal number of frames needed in one chunk.
        left_context (int): The number of frames required as left context.
        right_context (int): The number of frames required as right context.
        num_buffers (int): The number of parallel buffers.

    Example:

        >>> b = Buffer(3, 2, 4)
        >>> b.update(np.arange(20).reshape(10,2), 0, False)
        Chunk(data [(10, 2)], offset [0], is-last [False], left[0], right[4])

        array([[ 0,  1],
               [ 2,  3],
               [ 4,  5],
               [ 6,  7],
               [ 8,  9],
               [10, 11],
               [12, 13],
               [14, 15],
               [16, 17],
               [18, 19]])
        >>> b.update(np.arange(2).reshape(1,2), 10, False)
        None
        >>> b.update(np.arange(6).reshape(3,2), 11, False)
        Chunk(data [(10, 2)], offset [4], is-last [False], left[2], right[4])

        array([[ 8,  9],
               [10, 11],
               [12, 13],
               [14, 15],
               [16, 17],
               [18, 19],
               [ 0,  1],
               [ 0,  1],
               [ 2,  3],
               [ 4,  5]])
        >>> b.update(np.arange(2).reshape(1,2), 14, True)
        Chunk(data [(7, 2)], offset [8], is-last [True], left[2], right[0])

        array([[16, 17],
               [18, 19],
               [ 0,  1],
               [ 0,  1],
               [ 2,  3],
               [ 4,  5],
               [ 0,  1]])
    """

    def __init__(self, min_frames, left_context, right_context, num_buffers=1):
        if num_buffers < 1:
            raise ValueError('Number of buffers has to be a positive int (>= 1).')

        self.min_frames = min_frames
        self.left_context = left_context
        self.right_context = right_context
        self.num_buffers = num_buffers

        self.buffers = [None] * self.num_buffers
        self.buffers_full = [False] * self.num_buffers

        self.current_frame = 0
        self.current_left_context = 0

    def update(self, data, offset, is_last, buffer_index=0):
        """
        Update the buffer at the given index.

        Args:
            data (np.ndarray): The frames.
            offset (int): The index of the first frame in `data` within the sequence.
            is_last (bool): Whether this is the last block of frames in the sequence.
            buffer_index (int): The index of the buffer to update (< self.num_buffers).
        """
        if buffer_index >= self.num_buffers:
            raise ValueError('Expected buffer index < {} but got index {}.'.format(self.num_buffers, buffer_index))

        if self.buffers[buffer_index] is not None and self.buffers[buffer_index].shape[0] > 0:
            expected_next_frame = self.current_frame + self.buffers[buffer_index].shape[0]
            if expected_next_frame != offset:
                raise ValueError(
                    'There are missing frames. Last frame in buffer is {}. The passed frames start at {}.'.format(
                        expected_next_frame, offset))

            self.buffers[buffer_index] = np.vstack([self.buffers[buffer_index], data])
        else:
            self.buffers[buffer_index] = data

        self.buffers_full[buffer_index] = is_last

    def get(self):
        """
        Get a new chunk if available.

        Returns:
            Chunk or list: If enough frames are available a chunk is returned. Otherwise None.
                           If ``self.num_buffer >= 1`` a list instead of single chunk is returned.
        """
        chunk_size = self._smallest_buffer()
        all_full = self._all_full()

        if all_full:
            right_context = 0
            num_frames = chunk_size - self.current_left_context
        else:
            right_context = self.right_context
            num_frames = self.min_frames

        chunk_size_needed = num_frames + self.current_left_context + right_context

        if chunk_size >= chunk_size_needed:
            data = []
            keep_frames = self.left_context + self.right_context
            keep_from = max(0, chunk_size - keep_frames)

            for index in range(self.num_buffers):
                data.append(self.buffers[index][:chunk_size])
                self.buffers[index] = self.buffers[index][keep_from:]

            if self.num_buffers == 1:
                data = data[0]

            chunk = Chunk(data,
                          self.current_frame,
                          all_full,
                          self.current_left_context,
                          right_context)

            self.current_left_context = min(self.left_context, chunk_size)
            self.current_frame = max(self.current_frame + chunk_size - keep_frames, 0)

            return chunk

    def _smallest_buffer(self):
        """
        Get the size of the smallest buffer.
        """

        smallest = np.inf

        for buffer in self.buffers:
            if buffer is None:
                return 0
            elif buffer.shape[0] < smallest:
                smallest = buffer.shape[0]

        return smallest

    def _all_full(self):
        """
        Return True if all buffers are full (last frame added).
        """
        for is_full in self.buffers_full:
            if not is_full:
                return False

        return True


class Step(processing.Processor, metaclass=abc.ABCMeta):
    """
    This class is the base class for a step in a processing pipeline.

    It handles the procedure of executing the pipeline. It makes sure the steps are computed in the correct order.
    It also provides the correct inputs to every step.

    Every step has to provide a ``compute`` method which is the actual processing.

    Args:
        name (str, optional): A name for identifying the step.
    """

    def __init__(self, name=None, min_frames=1, left_context=0, right_context=0):
        self.graph = nx.DiGraph()
        self.name = name

        self.min_frames = min_frames
        self.left_context = left_context
        self.right_context = right_context

    def process_frames(self, data, sampling_rate, first_frame_index=0, last=False, utterance=None, corpus=None):
        """
        Execute the processing of this step and all dependent parent steps.
        """
        steps = nx.algorithms.dag.topological_sort(self.graph)
        step_results = {}

        for step in steps:
            parent_steps = [edge[0] for edge in self.graph.in_edges(step)]

            if len(parent_steps) == 0:
                res = step.compute(data, sampling_rate, first_frame_index, last=last, utterance=utterance,
                                   corpus=corpus)
            elif isinstance(step, Computation):
                parent_output = step_results[parent_steps[0]]
                res = step.compute(parent_output, sampling_rate, first_frame_index, last=last, utterance=utterance,
                                   corpus=corpus)
            else:
                # use step.parents to make sure the same order is kept as in the constructor of the reduction
                parent_outputs = [step_results[parent] for parent in step.parents]
                res = step.compute(parent_outputs, sampling_rate, first_frame_index, last=last, utterance=utterance,
                                   corpus=corpus)

            if step == self:
                return res
            else:
                step_results[step] = res

    @abc.abstractmethod
    def compute(self, data, sampling_rate, first_frame_index=0, last=False, corpus=None, utterance=None):
        pass


class Computation(Step, metaclass=abc.ABCMeta):
    """
    Base class for a computation step.

    Args:
        parent (Step, optional): The parent step this step depends on.
        name (str, optional): A name for identifying the step.
    """

    def __init__(self, parent=None, name=None, min_frames=1, left_context=0, right_context=0):
        super(Computation, self).__init__(name=name, min_frames=min_frames,
                                          left_context=left_context, right_context=right_context)

        self.graph.add_node(self)

        if parent is not None:
            self.graph.add_nodes_from(parent.graph.nodes)
            self.graph.add_edges_from(parent.graph.edges)
            self.graph.add_edge(parent, self)

    def __repr__(self) -> str:
        if self.name is None:
            return 'Computation'
        else:
            return self.name


class Reduction(Step, metaclass=abc.ABCMeta):
    """
    Base class for a reduction step.

    Args:
        parents (list): List of parent steps this step depends on.
        name (str, optional): A name for identifying the step.
    """

    def __init__(self, parents, name=None, min_frames=1, left_context=0, right_context=0):
        super(Reduction, self).__init__(name=name, min_frames=min_frames,
                                        left_context=left_context, right_context=right_context)

        self.parents = list(parents)
        self.graph.add_node(self)

        for index, parent in enumerate(parents):
            self.graph.add_nodes_from(parent.graph.nodes)
            self.graph.add_edges_from(parent.graph.edges)
            self.graph.add_edge(parent, self)

    def __repr__(self) -> str:
        if self.name is None:
            return 'Reduction'
        else:
            return self.name
