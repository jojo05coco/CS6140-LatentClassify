## Implements various file existence, read/write utilities

import os
import shutil

# Returns true if the file named 'name' exists
# Returns false otherwise
def file_exists(name):
    return os.path.isfile(name)

# Return true if the folder named 'name' exists
# Returns false otherwise
def folder_exists(name):
    return os.path.isdir(name)

# Deletes the folder named 'name' if exists
def delete_folder(name):

    if not folder_exists(name):
        return

    shutil.rmtree(name)

# Create a folder named 'name'
def create_folder(name):
    os.mkdir(name)
    assert folder_exists(name)
