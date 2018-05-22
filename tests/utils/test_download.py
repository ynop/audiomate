import os

import pytest
import requests_mock

from audiomate.utils import download

from tests import resources


@pytest.fixture()
def sample_zip_data():
    with open(resources.get_resource_path(['sample_files', 'zip_sample.zip']), 'rb') as f:
        return f.read()


def test_download_file(sample_zip_data, tmpdir):
    dl_path = 'http://some.url/thezipfile.zip'
    target_path = os.path.join(tmpdir.strpath, 'target.zip')

    with requests_mock.Mocker() as mock:
        mock.get(dl_path, content=sample_zip_data)

        download.download_file(dl_path, target_path)

    assert os.path.isfile(target_path)

    with open(target_path, 'rb') as f:
        assert f.read() == sample_zip_data
