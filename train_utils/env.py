from __future__ import annotations

from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def load_project_dotenv(dotenv_path: str | Path | None = None, *, override: bool = False) -> Path | None:
    """
    Load environment variables from a .env file.

    If `dotenv_path` is provided, it must exist and be a file.
    If omitted, the function searches from the current working directory upward.
    """
    if dotenv_path is None:
        found = find_dotenv(usecwd=True)
        if not found:
            return None
        path = Path(found)
    else:
        path = Path(dotenv_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Requested dotenv file does not exist: {path}")

    load_dotenv(dotenv_path=path, override=override)
    return path
