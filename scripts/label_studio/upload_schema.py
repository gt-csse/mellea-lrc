"""Upload the Label Studio labeling schema for citation annotation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from .upload_tasks import get_session

if TYPE_CHECKING:
    import requests

CONFIG_PATH = Path(__file__).with_name("label_studio_config.xml")


def upload_schema(
    session: requests.Session,
    ls_url: str,
    project_id: str,
    schema: str,
) -> dict[str, object]:
    """Update the configured Label Studio project with the labeling schema."""
    csrf = session.cookies.get("csrftoken")
    if not csrf:
        msg = "CSRF token not found in cookies"
        raise ValueError(msg)
    response = session.patch(
        f"{ls_url}/api/projects/{project_id}",
        json={"label_config": schema},
        headers={"X-CSRFToken": csrf, "Referer": ls_url},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def main() -> int:
    """Upload this directory's Label Studio XML config."""
    load_dotenv(dotenv_path=Path(".env"))
    ls_url = os.environ["LS_URL"]
    project_id = os.environ["LS_PROJECT_ID"]
    session = get_session(ls_url, os.environ["LS_EMAIL"], os.environ["LS_PASSWORD"])

    result = upload_schema(session, ls_url, project_id, CONFIG_PATH.read_text(encoding="utf-8"))
    print(f"Schema updated for project {result['id']}: {result['title']}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
