import os

import pytest
import requests_mock

from audiomate.utils import download

from tests import resources


@pytest.fixture()
def sample_zip_data():
    with open(resources.get_resource_path(['sample_files', 'zip_sample.zip']), 'rb') as f:
        return f.read()


@pytest.fixture()
def sample_zip_path():
    return resources.get_resource_path(['sample_files', 'zip_sample.zip'])


def test_download_file(sample_zip_data, tmpdir):
    dl_path = 'http://some.url/thezipfile.zip'
    target_path = os.path.join(tmpdir.strpath, 'target.zip')

    with requests_mock.Mocker() as mock:
        mock.get(dl_path, content=sample_zip_data)

        download.download_file(dl_path, target_path)

    assert os.path.isfile(target_path)

    with open(target_path, 'rb') as f:
        assert f.read() == sample_zip_data


def test_extract_zip(sample_zip_path, tmpdir):
    target_folder = tmpdir.strpath

    download.extract_zip(sample_zip_path, target_folder)

    assert os.path.isfile(os.path.join(target_folder, 'a.txt'))
    assert os.path.isfile(os.path.join(target_folder, 'data', 'dibsdadu.txt'))
    assert os.path.isfile(os.path.join(target_folder, 'data', 'babadu.txt'))
