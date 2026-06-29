"""Tests for strict project-local environment loading."""

from pathlib import Path

import pytest

from mellea_lrc.core.env import load_env_file, read_env_file


def test_read_env_file_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("MODEL=first\nMODEL=second\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"line 2: MODEL"):
        read_env_file(path)


def test_load_env_file_preserves_process_values_by_default(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("MODEL=file\nAPI_BASE=https://example.test/v1\n", encoding="utf-8")
    environ = {"MODEL": "process"}

    load_env_file(path, environ=environ)

    assert environ == {"MODEL": "process", "API_BASE": "https://example.test/v1"}


def test_load_env_file_can_explicitly_override_process_values(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("MODEL=file\n", encoding="utf-8")
    environ = {"MODEL": "process"}

    load_env_file(path, environ=environ, override=True)

    assert environ == {"MODEL": "file"}
