import numpy as np

from audiomate.processing import pipeline
from audiomate.processing.pipeline import base


class Multiply(pipeline.Computation):
    def __init__(self, factor, parent=None, name=None):
        super(Multiply, self).__init__(parent=parent, name=name)
        self.factor = factor

        self.frame_scale = 1.0
        self.hop_scale = 1.0

    def compute(self, chunk, sampling_rate, corpus=None, utterance=None):
        return chunk.data * self.factor

    def frame_transform_step(self, frame_size, hop_size):
        return frame_size * self.frame_scale, hop_size * self.hop_scale


class Add(pipeline.Computation):
    def __init__(self, value, parent=None, name=None):
        super(Add, self).__init__(parent=parent, name=name)
        self.value = value

        self.frame_scale = 1.0
        self.hop_scale = 1.0

    def compute(self, chunk, sampling_rate, corpus=None, utterance=None):
        return chunk.data + self.value

    def frame_transform_step(self, frame_size, hop_size):
        return frame_size * self.frame_scale, hop_size * self.hop_scale


class Concat(pipeline.Reduction):

    def __init__(self, parents, name=None, min_frames=1, left_context=0, right_context=0):
        super(Concat, self).__init__(parents, name=name, min_frames=min_frames,
                                     left_context=left_context, right_context=right_context)

        self.frame_scale = 1.0
        self.hop_scale = 1.0

    def compute(self, chunk, sampling_rate, corpus=None, utterance=None):
        return np.hstack(chunk.data)

    def frame_transform_step(self, frame_size, hop_size):
        return frame_size * self.frame_scale, hop_size * self.hop_scale


class StepDummy(pipeline.Computation):
    def __init__(self, parent=None, name=None, min_frames=0, left_context=0, right_context=0):
        super(StepDummy, self).__init__(parent=parent, name=name, min_frames=min_frames,
                                        left_context=left_context, right_context=right_context)

    def compute(self, chunk, sampling_rate, corpus=None, utterance=None):
        start = chunk.left_context
        end = chunk.data.shape[0] - chunk.right_context

        return chunk.data[start:end]


class TestStep:
    def test_process(self):
        add_a = Add(5)
        mul = Multiply(2, parent=add_a)
        add_b = Add(2)
        add_c = Add(3, parent=add_a)

        concat = Concat(parents=[mul, add_b, add_c])

        in_data = np.array([[0, 1, 2, 3]])
        out_data = concat.process_frames(in_data, 4)

        assert np.array_equal(out_data, np.array([[10, 12, 14, 16, 2, 3, 4, 5, 8, 9, 10, 11]]))

    def test_process_with_single_step_and_context(self):
        context = StepDummy(min_frames=2, left_context=2, right_context=1)

        out_data = context.process_frames(np.array([[0, 1, 2, 3]]), 4, offset=0, last=False)
        assert out_data is None

        out_data = context.process_frames(np.array([[4, 5, 6, 7]]), 4, offset=1, last=False)
        assert out_data is None

        out_data = context.process_frames(np.array([[8, 9, 10, 11]]), 4, offset=2, last=False)
        assert np.array_equal(out_data, np.array([[0, 1, 2, 3],
                                                  [4, 5, 6, 7]]))

        out_data = context.process_frames(np.array([[12, 13, 14, 15]]), 4, offset=3, last=True)
        assert np.array_equal(out_data, np.array([[8, 9, 10, 11],
                                                  [12, 13, 14, 15]]))

    def test_process_with_reduction_and_context(self):
        context = StepDummy(name='DummyContext', min_frames=2, left_context=2, right_context=1)
        add_a = Add(5, name='Add5')

        concat = Concat(parents=[context, add_a], name='Concat')

        out_data = concat.process_frames(np.array([[0, 1, 2, 3]]), 4, offset=0, last=False)
        assert out_data is None

        out_data = concat.process_frames(np.array([[4, 5, 6, 7]]), 4, offset=1, last=False)
        assert out_data is None

        out_data = concat.process_frames(np.array([[8, 9, 10, 11]]), 4, offset=2, last=False)
        assert np.array_equal(out_data, np.array([[0, 1, 2, 3, 5, 6, 7, 8],
                                                  [4, 5, 6, 7, 9, 10, 11, 12]]))

        out_data = concat.process_frames(np.array([[12, 13, 14, 15]]), 4, offset=3, last=True)
        assert np.array_equal(out_data, np.array([[8, 9, 10, 11, 13, 14, 15, 16],
                                                  [12, 13, 14, 15, 17, 18, 19, 20]]))

    def test_process_with_context(self):
        context = StepDummy(min_frames=2, left_context=2, right_context=1)
        add_a = Add(5)
        mul = Multiply(2, parent=add_a)
        add_b = Add(2, parent=context)
        add_c = Add(3, parent=add_a)

        concat = Concat(parents=[mul, add_b, add_c])

        out_data = concat.process_frames(np.array([[0, 1, 2, 3]]), 4, offset=0, last=False)
        assert out_data is None

        out_data = concat.process_frames(np.array([[4, 5, 6, 7]]), 4, offset=1, last=False)
        assert out_data is None

        out_data = concat.process_frames(np.array([[8, 9, 10, 11]]), 4, offset=2, last=False)
        assert np.array_equal(out_data, np.array([[10, 12, 14, 16, 2, 3, 4, 5, 8, 9, 10, 11],
                                                  [18, 20, 22, 24, 6, 7, 8, 9, 12, 13, 14, 15]]))

        out_data = concat.process_frames(np.array([[12, 13, 14, 15]]), 4, offset=3, last=True)
        assert np.array_equal(out_data, np.array([[26, 28, 30, 32, 10, 11, 12, 13, 16, 17, 18, 19],
                                                  [34, 36, 38, 40, 14, 15, 16, 17, 20, 21, 22, 23]]))

    def test_frame_transform(self):
        add_a = Add(5)
        mul = Multiply(2, parent=add_a)
        concat = Concat(parents=[add_a, mul])
        add_b = Add(3, parent=concat)

        add_a.frame_scale = 2.0
        add_a.hop_scale = 1.5

        add_b.frame_scale = 0.75
        add_b.hop_scale = 1.25

        tf_fs, tf_hs = add_b.frame_transform(200, 120)

        assert tf_fs == 300
        assert tf_hs == 225

    def test_frame_transform_not_mocked_step(self):
        add_a = Add(5)
        mul_a = StepDummy(parent=add_a)
        add_b = Add(3, parent=mul_a)
        concat = Concat(parents=[add_b, mul_a])

        mul_x = Multiply(3, parent=concat)
        mul_y = Multiply(4, parent=concat)

        mul_x.frame_scale = 2.0
        mul_x.hop_scale = 2.0

        mul_y.frame_scale = 2.0
        mul_y.hop_scale = 2.0

        concat_fin = Concat(parents=[mul_x, mul_y])

        tf_fs, tf_hs = concat_fin.frame_transform(200, 120)

        assert tf_fs == 400
        assert tf_hs == 240


class TestBuffer:

    def test_not_enough_frames_and_not_last_returns_none(self):
        buffer = base.Buffer(3, 0, 0)
        chunk = np.arange(4).reshape(2, 2)

        buffer.update(chunk, 0, False)
        res = buffer.get()

        assert res is None

    def test_not_enough_frames_and_is_last_returns_rest(self):
        buffer = base.Buffer(3, 0, 0)
        chunk = np.arange(4).reshape(2, 2)
        buffer.update(chunk, 0, True)
        res = buffer.get()

        assert np.allclose(chunk, res.data)
        assert res.offset == 0
        assert res.is_last
        assert res.left_context == 0
        assert res.right_context == 0

    def test_has_enough_returns_all(self):
        buffer = base.Buffer(3, 0, 0)
        chunk1 = np.arange(4).reshape(2, 2)
        chunk2 = np.arange(6).reshape(3, 2) + 4

        buffer.update(chunk1, 0, False)
        res = buffer.get()
        assert res is None

        buffer.update(chunk2, 2, False)
        res = buffer.get()
        assert np.allclose(np.vstack([chunk1, chunk2]), res.data)
        assert res.offset == 0
        assert not res.is_last
        assert res.left_context == 0
        assert res.right_context == 0

    def test_returns_correct_offset(self):
        buffer = base.Buffer(2, 0, 0)
        chunk1 = np.arange(4).reshape(2, 2)
        chunk2 = np.arange(6).reshape(3, 2) + 4

        buffer.update(chunk1, 0, False)
        res = buffer.get()
        assert res.offset == 0

        buffer.update(chunk2, 2, False)
        res = buffer.get()
        assert res.offset == 2

    def test_with_left_context(self):
        buffer = base.Buffer(2, 3, 0)
        chunk = np.arange(4).reshape(2, 2)

        buffer.update(chunk, 0, False)
        res = buffer.get()
        assert np.allclose(chunk, res.data)
        assert res.offset == 0
        assert not res.is_last
        assert res.left_context == 0
        assert res.right_context == 0

    def test_returns_correct_number_of_frames_with_left_context(self):
        buffer = base.Buffer(2, 3, 0)
        chunk1 = np.arange(8).reshape(4, 2)
        chunk2 = np.arange(6).reshape(3, 2) + 8

        buffer.update(chunk1, 0, False)
        res = buffer.get()
        assert np.allclose(chunk1, res.data)
        assert res.offset == 0
        assert not res.is_last
        assert res.left_context == 0
        assert res.right_context == 0

        buffer.update(chunk2, 4, False)
        res = buffer.get()
        assert np.allclose(np.vstack([chunk1[-3:], chunk2]), res.data)
        assert res.offset == 1
        assert not res.is_last
        assert res.left_context == 3
        assert res.right_context == 0

    def test_returns_correct_number_of_left_context(self):
        buffer = base.Buffer(2, 3, 0)
        chunk1 = np.arange(4).reshape(2, 2)
        chunk2 = np.arange(6).reshape(3, 2) + 4

        buffer.update(chunk1, 0, False)
        res = buffer.get()
        assert res.offset == 0
        assert not res.is_last
        assert res.left_context == 0
        assert res.right_context == 0

        buffer.update(chunk2, 2, False)
        res = buffer.get()
        assert res.offset == 0
        assert not res.is_last
        assert res.left_context == 2
        assert res.right_context == 0

    def test_dont_include_context_into_min_frames(self):
        buffer = base.Buffer(3, 1, 0)
        chunk1 = np.arange(8).reshape(4, 2)
        chunk2 = np.arange(4).reshape(2, 2) + 8
        chunk3 = np.arange(2).reshape(1, 2) + 11

        buffer.update(chunk1, 0, False)
        res = buffer.get()
        assert res.data.shape[0] == 4

        buffer.update(chunk2, 4, False)
        res = buffer.get()
        assert res is None

        buffer.update(chunk3, 6, False)
        res = buffer.get()
        assert res.data.shape[0] == 4
        assert res.offset == 3
        assert not res.is_last
        assert res.left_context == 1
        assert res.right_context == 0

    def test_not_enough_frames_and_is_last_returns_rest_with_left_context(self):
        buffer = base.Buffer(3, 2, 0)

        chunk1 = np.arange(8).reshape(4, 2)
        chunk2 = np.arange(4).reshape(2, 2) + 8

        buffer.update(chunk1, 0, False)
        res = buffer.get()
        assert np.allclose(chunk1, res.data)
        assert res.offset == 0
        assert not res.is_last
        assert res.left_context == 0
        assert res.right_context == 0

        buffer.update(chunk2, 4, True)
        res = buffer.get()
        assert np.allclose(np.vstack([chunk1[2:], chunk2]), res.data)
        assert res.offset == 2
        assert res.is_last
        assert res.left_context == 2
        assert res.right_context == 0

    def test_right_context(self):
        buffer = base.Buffer(2, 1, 2)

        chunk1 = np.arange(4).reshape(2, 2)
        chunk2 = np.arange(6).reshape(3, 2) + 4

        buffer.update(chunk1, 0, False)
        res = buffer.get()
        assert res is None

        buffer.update(chunk2, 2, False)
        res = buffer.get()
        assert np.allclose(np.vstack([chunk1, chunk2]), res.data)
        assert res.offset == 0
        assert not res.is_last
        assert res.left_context == 0
        assert res.right_context == 2

    def test_not_enough_frames_and_is_last_returns_rest_with_right_context(self):
        buffer = base.Buffer(10, 1, 2)

        chunk1 = np.arange(24).reshape(12, 2)
        chunk2 = np.arange(2).reshape(1, 2) + 24

        buffer.update(chunk1, 0, False)
        res = buffer.get()
        assert np.allclose(chunk1, res.data)
        assert res.offset == 0
        assert not res.is_last
        assert res.left_context == 0
        assert res.right_context == 2

        buffer.update(chunk2, 12, True)
        res = buffer.get()
        assert np.allclose(np.vstack([chunk1[9:], chunk2]), res.data)
        assert res.offset == 9
        assert res.is_last
        assert res.left_context == 1
        assert res.right_context == 0

    def test_multiple_buffers(self):
        buffer = base.Buffer(3, 2, 0, num_buffers=2)

        chunk1 = np.arange(8).reshape(4, 2)
        buffer.update(chunk1, 0, False, buffer_index=0)
        buffer.update(chunk1, 0, False, buffer_index=1)
        res = buffer.get()

        assert len(res.data) == 2
        assert np.allclose(chunk1, res.data[0])
        assert np.allclose(chunk1, res.data[1])
        assert res.offset == 0
        assert not res.is_last
        assert res.left_context == 0
        assert res.right_context == 0

        chunk2 = np.arange(4).reshape(2, 2) + 8
        buffer.update(chunk2, 4, True, buffer_index=0)
        res = buffer.get()

        assert res is None

        buffer.update(chunk2, 4, True, buffer_index=1)
        res = buffer.get()

        assert len(res.data) == 2
        assert np.allclose(np.vstack([chunk1[2:], chunk2]), res.data[0])
        assert np.allclose(np.vstack([chunk1[2:], chunk2]), res.data[1])
        assert res.offset == 2
        assert res.is_last
        assert res.left_context == 2
        assert res.right_context == 0

    def test_multiple_buffers_with_different_income_timings(self):
        buffer = base.Buffer(min_frames=1, left_context=0, right_context=0, num_buffers=2)

        buffer.update(np.array([[1, 2, 3, 4]]), 0, False, buffer_index=1)
        res = buffer.get()
        assert res is None

        buffer.update(np.array([[5, 6, 7, 8]]), 1, False, buffer_index=1)
        res = buffer.get()
        assert res is None

        buffer.update(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]), 0, False, buffer_index=0)
        buffer.update(np.array([[9, 10, 11, 12]]), 2, False, buffer_index=1)
        res = buffer.get()

        assert buffer.buffers[0].shape[0] == 0
        assert np.array_equal(buffer.buffers[1], np.array([[9, 10, 11, 12]]))

        assert len(res.data) == 2
        assert np.array_equal(res.data[0], np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))
        assert np.array_equal(res.data[1], np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
        assert not res.is_last
        assert res.offset == 0
        assert res.left_context == 0
        assert res.right_context == 0

        buffer.update(np.array([[8, 9, 10, 11], [12, 13, 14, 15]]), 2, True, buffer_index=0)
        buffer.update(np.array([[13, 14, 15, 16]]), 3, True, buffer_index=1)
        res = buffer.get()

        assert len(res.data) == 2
        assert np.array_equal(res.data[0], np.array([[8, 9, 10, 11], [12, 13, 14, 15]]))
        assert np.array_equal(res.data[1], np.array([[9, 10, 11, 12], [13, 14, 15, 16]]))
        assert res.is_last
        assert res.offset == 2
        assert res.left_context == 0
        assert res.right_context == 0
