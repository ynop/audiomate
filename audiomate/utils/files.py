import os
import shutil


def move_all_files_from_subfolders_to_top(folder_path, delete_subfolders=False):
    """
    Move all files/folder from all subfolders of `folder_path` on top into `folder_path`.

    Args:
        folder_path (str): Path of the folder.
        delete_subfolders (bool): If True the subfolders are deleted after all items are moved out of it.
    """
    for item in os.listdir(folder_path):
        sub_path = os.path.join(folder_path, item)

        if os.path.isdir(sub_path):

            for sub_item in os.listdir(sub_path):
                src = os.path.join(sub_path, sub_item)
                target = os.path.join(folder_path, sub_item)
                shutil.move(src, target)

            if delete_subfolders:
                shutil.rmtree(sub_path)
