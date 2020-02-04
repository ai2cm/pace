import os
import fsspec


def get_fs(path: str) -> fsspec.AbstractFileSystem:
    """Return the fsspec filesystem required to handle a given path."""
    if path.startswith("gs://"):
        return fsspec.filesystem("gs")
    else:
        return fsspec.filesystem("file")


def is_file(filename):
    return get_fs(filename).isfile(filename)


def open(filename, *args, **kwargs):
    fs = get_fs(filename)
    return fs.open(filename, *args, **kwargs)
