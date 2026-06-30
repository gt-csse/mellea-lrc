"""Upload text documents to Label Studio with extraction pre-annotations."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
from mellea_lrc.core.env import load_env_file
from charset_normalizer import from_path

from .pre_annotate import build_task_payload


def get_session(ls_url: str, email: str, password: str) -> requests.Session:
    """Authenticate with Label Studio and return a session with CSRF cookies."""
    session = requests.Session()
    session.get(f"{ls_url}/user/login", timeout=30)
    csrf = session.cookies.get("csrftoken")
    if not csrf:
        msg = "CSRF token not found in cookies"
        raise ValueError(msg)
    session.post(
        f"{ls_url}/user/login",
        data={"email": email, "password": password, "csrfmiddlewaretoken": csrf},
        headers={"Referer": f"{ls_url}/user/login", "X-CSRFToken": csrf},
        timeout=30,
    )
    return session


def upload_task(
    session: requests.Session,
    ls_url: str,
    project_id: str,
    text: str,
    *,
    source_path: str | None = None,
) -> dict[str, object]:
    """Upload one Label Studio task with a pre-annotation prediction."""
    task_payload = build_task_payload(text, source_path=source_path)
    csrf = session.cookies.get("csrftoken")
    if not csrf:
        msg = "CSRF token not found in cookies"
        raise ValueError(msg)
    response = session.post(
        f"{ls_url}/api/projects/{project_id}/import",
        json=[task_payload],
        headers={"X-CSRFToken": csrf, "Referer": ls_url},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def main(paths: list[str] | None = None) -> int:
    """Upload one or more text files to the configured Label Studio project."""
    load_env_file(Path(".env"))
    document_paths = paths if paths is not None else sys.argv[1:]
    if not document_paths:
        print(  # noqa: T201
            "Usage: uv run --group label-studio "
            "python -m scripts.label_studio.cli upload-tasks <file1.txt> [file2.txt ...]"
        )
        return 1

    session = get_session(
        os.environ["LS_URL"],
        os.environ["LS_EMAIL"],
        os.environ["LS_PASSWORD"],
    )
    project_id = os.environ["LS_PROJECT_ID"]

    for document_path in document_paths:
        text = str(from_path(Path(document_path)).best())
        result = upload_task(
            session,
            os.environ["LS_URL"],
            project_id,
            text,
            source_path=document_path,
        )
        print(f"{document_path}: {result['task_count']} task, {result['prediction_count']} prediction")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
