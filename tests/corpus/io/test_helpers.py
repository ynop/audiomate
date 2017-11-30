import unittest

from pingu.corpus.io import *


class HelpersTest(unittest.TestCase):

    def test_all_loaders_registered(self):
        expected_loaders = CorpusLoader.__subclasses__()
        actual_loaders = available_loaders()

        self.assertEqual(len(expected_loaders), len(actual_loaders),
                         'Number of registered loaders does not match number of present loaders')

        for expected_loader in expected_loaders:
            self.assertIn(expected_loader, actual_loaders.values(), 'Loader not registered')
            self.assertIn(expected_loader.type(), actual_loaders.keys(), 'Loader not available under its type()')

    def test_all_loaders_creatable(self):
        expected_loaders = CorpusLoader.__subclasses__()

        for expected_loader in expected_loaders:
            self.assertIsInstance(create_loader_of_type(expected_loader.type()), expected_loader)

    def test_unknown_loader_creation_throws(self):
        with self.assertRaises(UnknownLoaderException, msg='Unknown loader: does_not_exist'):
            create_loader_of_type('does_not_exist')
