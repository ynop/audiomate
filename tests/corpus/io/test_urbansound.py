import os

import pytest

from audiomate.corpus import io
from tests import resources


@pytest.fixture
def reader():
    return io.Urbansound8kReader()


@pytest.fixture
def data_path():
    return resources.sample_corpus_path('urbansound8k')


class TestUrbansound8kReader:

    def test_load_files(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_files == 5

        assert ds.files['100032-3-0-0'].idx == '100032-3-0-0'
        assert ds.files['100032-3-0-0'].path == os.path.join(data_path, 'audio', 'fold5', '100032-3-0-0.wav')

        assert ds.files['100263-2-0-117'].idx == '100263-2-0-117'
        assert ds.files['100263-2-0-117'].path == os.path.join(data_path, 'audio', 'fold5', '100263-2-0-117.wav')

        assert ds.files['145612-6-3-0'].idx == '145612-6-3-0'
        assert ds.files['145612-6-3-0'].path == os.path.join(data_path, 'audio', 'fold8', '145612-6-3-0.wav')

        assert ds.files['145683-6-5-0'].idx == '145683-6-5-0'
        assert ds.files['145683-6-5-0'].path == os.path.join(data_path, 'audio', 'fold9', '145683-6-5-0.wav')

        assert ds.files['79377-9-0-4'].idx == '79377-9-0-4'
        assert ds.files['79377-9-0-4'].path == os.path.join(data_path, 'audio', 'fold2', '79377-9-0-4.wav')

    def test_load_utterances(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_utterances == 5

        assert ds.utterances['100032-3-0-0'].idx == '100032-3-0-0'
        assert ds.utterances['100032-3-0-0'].file.idx == '100032-3-0-0'
        assert ds.utterances['100032-3-0-0'].issuer is None
        assert ds.utterances['100032-3-0-0'].start == 0
        assert ds.utterances['100032-3-0-0'].end == -1

        assert ds.utterances['100263-2-0-117'].idx == '100263-2-0-117'
        assert ds.utterances['100263-2-0-117'].file.idx == '100263-2-0-117'
        assert ds.utterances['100263-2-0-117'].issuer is None
        assert ds.utterances['100263-2-0-117'].start == 0
        assert ds.utterances['100263-2-0-117'].end == -1

        assert ds.utterances['145612-6-3-0'].idx == '145612-6-3-0'
        assert ds.utterances['145612-6-3-0'].file.idx == '145612-6-3-0'
        assert ds.utterances['145612-6-3-0'].issuer is None
        assert ds.utterances['145612-6-3-0'].start == 0
        assert ds.utterances['145612-6-3-0'].end == -1

        assert ds.utterances['145683-6-5-0'].idx == '145683-6-5-0'
        assert ds.utterances['145683-6-5-0'].file.idx == '145683-6-5-0'
        assert ds.utterances['145683-6-5-0'].issuer is None
        assert ds.utterances['145683-6-5-0'].start == 0
        assert ds.utterances['145683-6-5-0'].end == -1

        assert ds.utterances['79377-9-0-4'].idx == '79377-9-0-4'
        assert ds.utterances['79377-9-0-4'].file.idx == '79377-9-0-4'
        assert ds.utterances['79377-9-0-4'].issuer is None
        assert ds.utterances['79377-9-0-4'].start == 0
        assert ds.utterances['79377-9-0-4'].end == -1

    def test_load_issuers(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_issuers == 0

    def test_load_labels(self, reader, data_path):
        ds = reader.load(data_path)

        assert len(ds.utterances['100032-3-0-0'].label_lists) == 1
        assert len(ds.utterances['100032-3-0-0'].label_lists['default']) == 1
        assert ds.utterances['100032-3-0-0'].label_lists['default'][0].value == 'dog_bark'
        assert ds.utterances['100032-3-0-0'].label_lists['default'][0].start == 0
        assert ds.utterances['100032-3-0-0'].label_lists['default'][0].end == -1

        assert len(ds.utterances['100263-2-0-117'].label_lists) == 1
        assert len(ds.utterances['100263-2-0-117'].label_lists['default']) == 1
        assert ds.utterances['100263-2-0-117'].label_lists['default'][0].value == 'children_playing'
        assert ds.utterances['100263-2-0-117'].label_lists['default'][0].start == 0
        assert ds.utterances['100263-2-0-117'].label_lists['default'][0].end == -1

        assert len(ds.utterances['145612-6-3-0'].label_lists) == 1
        assert len(ds.utterances['145612-6-3-0'].label_lists['default']) == 1
        assert ds.utterances['145612-6-3-0'].label_lists['default'][0].value == 'gun_shot'
        assert ds.utterances['145612-6-3-0'].label_lists['default'][0].start == 0
        assert ds.utterances['145612-6-3-0'].label_lists['default'][0].end == -1

        assert len(ds.utterances['145683-6-5-0'].label_lists) == 1
        assert len(ds.utterances['145683-6-5-0'].label_lists['default']) == 1
        assert ds.utterances['145683-6-5-0'].label_lists['default'][0].value == 'gun_shot'
        assert ds.utterances['145683-6-5-0'].label_lists['default'][0].start == 0
        assert ds.utterances['145683-6-5-0'].label_lists['default'][0].end == -1

        assert len(ds.utterances['79377-9-0-4'].label_lists) == 1
        assert len(ds.utterances['79377-9-0-4'].label_lists['default']) == 1
        assert ds.utterances['79377-9-0-4'].label_lists['default'][0].value == 'street_music'
        assert ds.utterances['79377-9-0-4'].label_lists['default'][0].start == 0
        assert ds.utterances['79377-9-0-4'].label_lists['default'][0].end == -1

    def test_load_fold_subsets(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_subviews == 4

        assert ds.subviews['fold2'].num_utterances == 1
        assert ds.subviews['fold5'].num_utterances == 2
        assert ds.subviews['fold8'].num_utterances == 1
        assert ds.subviews['fold9'].num_utterances == 1
