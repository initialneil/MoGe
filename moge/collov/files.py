import contextlib
import os

from datetime import datetime
from pathlib import Path


class WorkingDirectory(contextlib.ContextDecorator):
    """
    A context manager and decorator for temporarily changing the working directory.

    This class allows for the temporary change of the working directory using a context manager or decorator.
    It ensures that the original working directory is restored after the context or decorated function completes.

    Attributes:
        dir (Path): The new directory to switch to.
        cwd (Path): The original current working directory before the switch.

    Methods:
        __enter__: Changes the current directory to the specified directory.
        __exit__: Restores the original working directory on context exit.

    Examples:
        Using as a context manager:
        >>> with WorkingDirectory('/path/to/new/dir'):
        >>> # Perform operations in the new directory
        >>>     pass

        Using as a decorator:
        >>> @WorkingDirectory('/path/to/new/dir')
        >>> def some_function():
        >>> # Perform operations in the new directory
        >>>     pass
    """

    def __init__(self, new_dir):
        """Sets the working directory to 'new_dir' upon instantiation for use with context managers or decorators."""
        self.dir = new_dir  # new dir
        self.cwd = Path.cwd().resolve()  # current dir

    def __enter__(self):
        """Changes the current working directory to the specified directory upon entering the context."""
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa
        """Restores the original working directory when exiting the context."""
        os.chdir(self.cwd)


def get_all_subdirectories(dir_path, recursive=False):
    """
    Scan a directory to find all subdirectories.
    Args:
        dir_path (str | :obj:`Path`): Path of the directory.
        recursive (bool, optional): If set to True, recursively scan the directory.
        Defaults to False.
    Returns:
        A generator for all subdirectories with relative paths.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')
    root = dir_path

    def _scandir_subdirs(dir_path, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_dir():
                rel_path = os.path.relpath(entry.path, root)
                yield rel_path
                if recursive:  # scan recursively if entry.path is a directory
                    yield from _scandir_subdirs(entry.path, recursive)

    return _scandir_subdirs(dir_path, recursive)


def scandir(dir_path, suffix=None, recursive=False, case_sensitive=True):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str | :obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Defaults to None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Defaults to False.
        case_sensitive (bool, optional) : If set to False, ignore the case of
            suffix. Defaults to True.

    Returns:
        A generator for all the interested files with relative paths.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    if suffix is not None and not case_sensitive:
        suffix = (
            suffix.lower()
            if isinstance(suffix, str)
            else tuple(item.lower() for item in suffix)
        )

    root = dir_path

    def _scandir(dir_path, suffix, recursive, case_sensitive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                rel_path = os.path.relpath(entry.path, root)
                _rel_path = rel_path if case_sensitive else rel_path.lower()
                if suffix is None or _rel_path.endswith(suffix):
                    yield rel_path
            elif recursive and os.path.isdir(entry.path):
                # scan recursively if entry.path is a directory
                yield from _scandir(entry.path, suffix, recursive, case_sensitive)

    return _scandir(dir_path, suffix, recursive, case_sensitive)


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == "":
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def file_age(path=__file__):
    """Return days since the last modification of the specified file."""
    dt = datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime)  # delta
    return dt.days  # + dt.seconds / 86400  # fractional days


def file_date(path=__file__):
    """Returns the file modification date in 'YYYY-M-D' format."""
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f"{t.year}-{t.month}-{t.day}"


def file_size(path, unit="MB") -> float:
    """Returns the size of a file or directory in megabytes (MB)."""
    if isinstance(path, (str, Path)):
        bit_index = {"B": 0, "KB": 10, "MB": 20, "GB": 30, "TB": 40}
        bit_offset = 1 << bit_index[unit.upper()]
        # mb = 1 << 20  # bytes to MiB (1024 ** 2)
        path = Path(path)
        if path.is_file():
            return path.stat().st_size / bit_offset
        elif path.is_dir():
            return (
                sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())
                / bit_offset
            )
    return 0.0
