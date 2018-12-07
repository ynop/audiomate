from audiomate.corpus.io import CorpusDownloader, available_downloaders, create_downloader_of_type
from audiomate.corpus.io import CorpusReader, available_readers, create_reader_of_type
from audiomate.corpus.io import CorpusWriter, available_writers, create_writer_of_type
from audiomate.corpus.io import UnknownDownloaderException, UnknownReaderException, UnknownWriterException
from audiomate.corpus.io.downloader import ArchiveDownloader

import pytest


def test_all_downloaders_registered():
    expected_downloaders = CorpusDownloader.__subclasses__()
    expected_downloaders.remove(ArchiveDownloader)
    actual_downloaders = available_downloaders()

    assert len(expected_downloaders) == len(actual_downloaders), \
        'Number of registered downloaders does not match number of present downloaders'

    for expected_downloader in expected_downloaders:
        assert expected_downloader in actual_downloaders.values(), \
            'Downloader not registered'
        assert expected_downloader.type() in actual_downloaders.keys(), \
            'Downloader not available under its type()'


def test_all_downloaders_creatable():
    expected_downloaders = CorpusDownloader.__subclasses__()
    expected_downloaders.remove(ArchiveDownloader)

    for expected_downloader in expected_downloaders:
        assert isinstance(
            create_downloader_of_type(expected_downloader.type()),
            expected_downloader
        )


def test_unknown_downloader_creation_throws():
    with pytest.raises(UnknownDownloaderException, message='Unknown downloader: does_not_exist'):
        create_downloader_of_type('does_not_exist')


def test_all_readers_registered():
    expected_readers = CorpusReader.__subclasses__()
    actual_readers = available_readers()

    assert len(expected_readers) == len(actual_readers), \
        'Number of registered readers does not match number of present readers'

    for expected_reader in expected_readers:
        assert expected_reader in actual_readers.values(), \
            'Reader not registered'
        assert expected_reader.type() in actual_readers.keys(), \
            'Reader not available under its type()'


def test_all_readers_creatable():
    expected_readers = CorpusReader.__subclasses__()

    for expected_reader in expected_readers:
        assert isinstance(create_reader_of_type(expected_reader.type()), expected_reader)


def test_unknown_reader_creation_throws():
    with pytest.raises(UnknownReaderException, message='Unknown reader: does_not_exist'):
        create_reader_of_type('does_not_exist')


def test_all_writers_registered():
    expected_writers = CorpusWriter.__subclasses__()
    actual_writers = available_writers()

    assert len(expected_writers) == len(actual_writers), \
        'Number of registered writers does not match number of present writers'

    for expected_writer in expected_writers:
        assert expected_writer in actual_writers.values(), \
            'Writer not registered'
        assert expected_writer.type() in actual_writers.keys(), \
            'Writer not available under its type()'


def test_all_writers_creatable():
    expected_writers = CorpusWriter.__subclasses__()

    for expected_writer in expected_writers:
        assert isinstance(create_writer_of_type(expected_writer.type()), expected_writer)


def test_unknown_writer_creation_throws():
    with pytest.raises(UnknownWriterException, message='Unknown writer: does_not_exist'):
        create_writer_of_type('does_not_exist')
