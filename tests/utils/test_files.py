import os

from audiomate.utils import files


def test_move_all_files_from_subfolders_to_top(tmpdir):
    base_path = tmpdir.strpath

    os.makedirs(os.path.join(base_path, 'a', 'asub'))
    os.makedirs(os.path.join(base_path, 'b'))

    open(os.path.join(base_path, 'a', '1.txt'), 'w').close()
    open(os.path.join(base_path, 'a', 'asub', '3.txt'), 'w').close()
    open(os.path.join(base_path, 'b', '2.txt'), 'w').close()

    files.move_all_files_from_subfolders_to_top(base_path)

    assert len(os.listdir(base_path)) == 5

    assert os.path.isdir(os.path.join(base_path, 'a'))
    assert os.path.isdir(os.path.join(base_path, 'b'))
    assert os.path.isdir(os.path.join(base_path, 'asub'))
    assert os.path.isfile(os.path.join(base_path, '1.txt'))
    assert os.path.isfile(os.path.join(base_path, '2.txt'))


def test_move_all_files_from_subfolders_to_top_and_delete_subfolders(tmpdir):
    base_path = tmpdir.strpath

    os.makedirs(os.path.join(base_path, 'a', 'asub'))
    os.makedirs(os.path.join(base_path, 'b'))

    open(os.path.join(base_path, 'a', '1.txt'), 'w').close()
    open(os.path.join(base_path, 'a', 'asub', '3.txt'), 'w').close()
    open(os.path.join(base_path, 'b', '2.txt'), 'w').close()

    files.move_all_files_from_subfolders_to_top(base_path, delete_subfolders=True)

    assert len(os.listdir(base_path)) == 3

    assert os.path.isdir(os.path.join(base_path, 'asub'))
    assert os.path.isfile(os.path.join(base_path, '1.txt'))
    assert os.path.isfile(os.path.join(base_path, '2.txt'))
