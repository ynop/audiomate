import unittest

from audiomate.corpus.io import CorpusDownloader, available_downloaders, create_downloader_of_type
from audiomate.corpus.io import CorpusReader, available_readers, create_reader_of_type
from audiomate.corpus.io import CorpusWriter, available_writers, create_writer_of_type
from audiomate.corpus.io import UnknownDownloaderException, UnknownReaderException, UnknownWriterException


class HelpersTest(unittest.TestCase):

    def test_all_downloaders_registered(self):
        expected_downloaders = CorpusDownloader.__subclasses__()
        actual_downloaders = available_downloaders()

        self.assertEqual(len(expected_downloaders), len(actual_downloaders),
                         'Number of registered downloaders does not match number of present downloaders')

        for expected_downloader in expected_downloaders:
            self.assertIn(expected_downloader, actual_downloaders.values(), 'Downloader not registered')
            self.assertIn(expected_downloader.type(), actual_downloaders.keys(),
                          'Downloader not available under its type()')

    def test_all_downloaders_creatable(self):
        expected_downloaders = CorpusDownloader.__subclasses__()

        for expected_downloader in expected_downloaders:
            self.assertIsInstance(create_downloader_of_type(expected_downloader.type()), expected_downloader)

    def test_unknown_downloader_creation_throws(self):
        with self.assertRaises(UnknownDownloaderException, msg='Unknown downloader: does_not_exist'):
            create_downloader_of_type('does_not_exist')

    def test_all_readers_registered(self):
        expected_readers = CorpusReader.__subclasses__()
        actual_readers = available_readers()

        self.assertEqual(len(expected_readers), len(actual_readers),
                         'Number of registered readers does not match number of present readers')

        for expected_reader in expected_readers:
            self.assertIn(expected_reader, actual_readers.values(), 'Reader not registered')
            self.assertIn(expected_reader.type(), actual_readers.keys(),
                          'Reader not available under its type()')

    def test_all_readers_creatable(self):
        expected_readers = CorpusReader.__subclasses__()

        for expected_reader in expected_readers:
            self.assertIsInstance(create_reader_of_type(expected_reader.type()), expected_reader)

    def test_unknown_reader_creation_throws(self):
        with self.assertRaises(UnknownReaderException, msg='Unknown reader: does_not_exist'):
            create_reader_of_type('does_not_exist')

    def test_all_writers_registered(self):
        expected_writers = CorpusWriter.__subclasses__()
        actual_writers = available_writers()

        self.assertEqual(len(expected_writers), len(actual_writers),
                         'Number of registered writers does not match number of present writers')

        for expected_writer in expected_writers:
            self.assertIn(expected_writer, actual_writers.values(), 'Writer not registered')
            self.assertIn(expected_writer.type(), actual_writers.keys(),
                          'Writer not available under its type()')

    def test_all_writers_creatable(self):
        expected_writers = CorpusWriter.__subclasses__()

        for expected_writer in expected_writers:
            self.assertIsInstance(create_writer_of_type(expected_writer.type()), expected_writer)

    def test_unknown_writer_creation_throws(self):
        with self.assertRaises(UnknownWriterException, msg='Unknown writer: does_not_exist'):
            create_writer_of_type('does_not_exist')
