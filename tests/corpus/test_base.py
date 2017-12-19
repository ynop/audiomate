import unittest

from tests import resources


class CorpusViewTest(unittest.TestCase):
    def setUp(self):
        self.ds = resources.create_multi_label_corpus()

    def test_all_label_values(self):
        assert self.ds.all_label_values() == set(['music', 'speech'])

    def test_label_count(self):
        assert self.ds.label_count() == {'music': 11, 'speech': 7}
