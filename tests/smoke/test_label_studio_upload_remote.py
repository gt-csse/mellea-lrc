"""Remote smoke test for Label Studio upload with extraction pre-annotations."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = pytest.mark.remote_smoke

REQUIRED_ENV_KEYS = ("LS_URL", "LS_EMAIL", "LS_PASSWORD", "LS_PROJECT_ID")


def test_label_studio_upload_task_with_extraction_prediction() -> None:
    pytest.importorskip("dotenv", reason="Install with `uv sync --group label-studio`.")
    from scripts.label_studio.upload_tasks import get_session, upload_task

    config = _label_studio_config_from_env_file()
    session = get_session(config["LS_URL"], config["LS_EMAIL"], config["LS_PASSWORD"])
    smoke_id = uuid4().hex
    text = f"Smoke {smoke_id}: Brown v. Board, 347 U.S. 483."

    result = upload_task(
        session,
        config["LS_URL"],
        config["LS_PROJECT_ID"],
        text,
        source_path=f"smoke://label-studio-upload/{smoke_id}.txt",
    )

    assert result["task_count"] == 1
    assert result["prediction_count"] == 1


def _label_studio_config_from_env_file() -> dict[str, str]:
    values = {**os.environ, **_read_env_file(Path(".env"))}
    missing = [key for key in REQUIRED_ENV_KEYS if not values.get(key)]
    if missing:
        pytest.skip(f"Set {', '.join(missing)} in .env to run Label Studio upload smoke test.")
    return {key: values[key].rstrip("/") if key == "LS_URL" else values[key] for key in REQUIRED_ENV_KEYS}


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        pytest.skip("Create .env to run Label Studio upload smoke test.")

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values
