from pathlib import Path
import os
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_utils.env import load_project_dotenv


def test_load_project_dotenv_from_explicit_path(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("TEST_DOTENV_KEY=hello\n", encoding="utf-8")
    monkeypatch.delenv("TEST_DOTENV_KEY", raising=False)

    loaded_path = load_project_dotenv(env_path)
    assert loaded_path == env_path
    assert os.getenv("TEST_DOTENV_KEY") == "hello"


def test_load_project_dotenv_returns_none_when_not_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loaded_path = load_project_dotenv()
    assert loaded_path is None


def test_load_project_dotenv_fails_for_missing_explicit_path(tmp_path):
    missing = tmp_path / ".env.missing"
    with pytest.raises(FileNotFoundError):
        load_project_dotenv(missing)
