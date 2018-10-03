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

    If the implementation of a step does change the frame or hop-size,
    it is expected to provide a transform via the ``frame_transform_step`` method.
    Frame-size and hop-size are measured in samples regarding the original audio signal (or simply its sampling rate).

    Args:
        name (str, optional): A name for identifying the step.
    """

    def __init__(self, name=None, min_frames=1, left_context=0, right_context=0):
        self.graph = nx.DiGraph()
        self.name = name

        self.min_frames = min_frames
        self.left_context = left_context
        self.right_context = right_context

        self.steps_sorted = []
        self.buffers = {}
        self.target_buffers = {}

    def process_frames(self, data, sampling_rate, offset=0, last=False, utterance=None, corpus=None):
        """
        Execute the processing of this step and all dependent parent steps.
        """

        if offset == 0:
            self.steps_sorted = list(nx.algorithms.dag.topological_sort(self.graph))
            self._create_buffers()
            self._define_output_buffers()

        # Update buffers with input data
        self._update_buffers(None, data, offset, last)

        # Go through the ordered (by dependencies) steps
        for step in self.steps_sorted:

            chunk = self.buffers[step].get()

            if chunk is not None:
                res = step.compute(chunk, sampling_rate, utterance=utterance, corpus=corpus)

                # If step is self, we know its the last step so return the data
                if step == self:
                    return res

                # Otherwise update buffers of child steps
                else:
                    self._update_buffers(step, res, chunk.offset + chunk.left_context, chunk.is_last)

    def frame_transform(self, frame_size, hop_size):
        parent_steps = self._parent_steps(self)

        if len(parent_steps) > 0:
            parent_frame_size, parent_hop_size = parent_steps[0].frame_transform(frame_size, hop_size)

            if len(parent_steps) > 1:
                # If there are multiple parents, we ensure that the frame-size is equal from all parents
                for i in range(1, len(parent_steps)):
                    next_fs, next_hs = parent_steps[i].frame_transform(frame_size, hop_size)

                    if next_fs != parent_frame_size:
                        raise ValueError('Frame-size from different differs!')

                    if next_hs != parent_hop_size:
                        raise ValueError('Hop-size from different differs!')

            return self.frame_transform_step(parent_frame_size, parent_hop_size)

        return self.frame_transform_step(frame_size, hop_size)

    @abc.abstractmethod
    def compute(self, chunk, sampling_rate, corpus=None, utterance=None):
        """
        Do the computation of the step. If the step uses context, the result has to be returned without context.

        Args:
            chunk (Chunk): The chunk containing data and info about context, offset, ...
            sampling_rate (int): The sampling rate of the underlying signal.
            corpus (Corpus): The corpus the data is from, if available.
            utterance (Utterance): The utterance the data is from, if available.

        Returns:
            np.ndarray: The array of processed frames, without context.
        """
        pass

    def frame_transform_step(self, frame_size, hop_size):
        """
        If the processor changes the number of samples that build up a frame or
        the number of samples between two consecutive frames (hop-size),
        this function needs transform the original frame- and/or hop-size.

        This is used to store the frame-size and hop-size in a feature-container.
        In the end one can calculate start and end time of a frame with this information.

        By default it is assumed that the processor doesn't change the frame-size and the hop-size.

        Note:
            This function is simply for this step, whereas ``frame_transform()``
            computes the transformation for the whole pipeline.

        Args:
            frame_size (int): The original frame-size.
            hop_size (int): The original hop-size.

        Returns:
            tuple: The (frame-size, hop-size) after processing.
        """
        return frame_size, hop_size

    def _update_buffers(self, from_step, data, offset, is_last):
        """
        Update the buffers of all steps that need data from ``from_step``.
        If ``from_step`` is None it means the data is the input data.
        """

        for to_step, buffer in self.target_buffers[from_step]:
            parent_index = 0

            # if there multiple inputs we have to get the correct index, to keep the ordering
            if isinstance(to_step, Reduction):
                parent_index = to_step.parents.index(from_step)

            buffer.update(data, offset, is_last, buffer_index=parent_index)

    def _define_output_buffers(self):
        """
        Prepare a dictionary so we know what buffers have to be update with the the output of every step.
        """

        # First define buffers that need input data
        self.target_buffers = {
            None: [(step, self.buffers[step]) for step in self._get_input_steps()]
        }

        # Go through all steps and append the buffers of their child nodes
        for step in self.steps_sorted:
            if step != self:
                child_steps = [edge[1] for edge in self.graph.out_edges(step)]
                self.target_buffers[step] = [(child_step, self.buffers[child_step]) for child_step in child_steps]

    def _get_input_steps(self):
        """
        Search and return all steps that have no parents. These are the steps that are get the input data.
        """
        input_steps = []

        for step in self.steps_sorted:
            parent_steps = self._parent_steps(step)

            if len(parent_steps) == 0:
                input_steps.append(step)

        return input_steps

    def _create_buffers(self):
        """
        Create a buffer for every step in the pipeline.
        """

        self.buffers = {}

        for step in self.graph.nodes():
            num_buffers = 1

            if isinstance(step, Reduction):
                num_buffers = len(step.parents)

            self.buffers[step] = Buffer(step.min_frames, step.left_context, step.right_context, num_buffers)

        return self.buffers

    def _parent_steps(self, step):
        """ Return a list of all parent steps. """
        return [edge[0] for edge in self.graph.in_edges(step)]


class Computation(Step, metaclass=abc.ABCMeta):
    """
    Base class for a computation step.
    To implement a computation step for pipeline the ``compute`` method has to be implemented.
    This method gets the frames from its parent step including context frames if defined.
    It has to return the same number of frames but without context frames.

    Args:
        parent (Step, optional): The parent step this step depends on.
        name (str, optional): A name for identifying the step.
    """

    def __init__(self, parent=None, name=None, min_frames=1, left_context=0, right_context=0):
        super(Computation, self).__init__(name=name, min_frames=min_frames,
                                          left_context=left_context, right_context=right_context)

        self.graph.add_node(self)

        if parent is not None:
            self.graph.add_nodes_from(parent.graph.nodes())
            self.graph.add_edges_from(parent.graph.edges())
            self.graph.add_edge(parent, self)

    def __repr__(self) -> str:
        if self.name is None:
            return 'Computation'
        else:
            return self.name


class Reduction(Step, metaclass=abc.ABCMeta):
    """
    Base class for a reduction step.
    It gets the frames of all its parent steps as a list.
    It has to return a single chunk of frames.

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
            self.graph.add_nodes_from(parent.graph.nodes())
            self.graph.add_edges_from(parent.graph.edges())
            self.graph.add_edge(parent, self)

    def __repr__(self) -> str:
        if self.name is None:
            return 'Reduction'
        else:
            return self.name
