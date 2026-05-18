import pathlib


def find_idea_root(start_path: pathlib.Path | str = ".") -> pathlib.Path | None:
    """
    Walks up directories from start_path until a directory containing a '.idea' folder is found.

    Args:
        start_path: The directory to start the search from. Defaults to the current directory.

    Returns:
        The pathlib.Path of the directory containing '.idea', or None if not found.
    """
    current_path = pathlib.Path(start_path).resolve()

    # Iterate up through parents
    # .parents includes all parent directories, but not the directory itself.
    # We should check the current directory first.

    for path in [current_path] + list(current_path.parents):
        if (path / ".idea").is_dir():
            return path

    return None
