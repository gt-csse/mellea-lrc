import os
import requests
from dotenv import load_dotenv

from scripts.annotation.upload_tasks import get_session


def upload_schema(s: requests.Session, ls_url: str, project_id: str, schema: str) -> dict:
    csrf = s.cookies["csrftoken"]
    r = s.patch(
        f"{ls_url}/api/projects/{project_id}",
        json={"label_config": schema},
        headers={"X-CSRFToken": csrf, "Referer": ls_url},
    )
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    load_dotenv()
    ls_url     = os.environ["LS_URL"]
    email      = os.environ["LS_EMAIL"]
    password   = os.environ["LS_PASSWORD"]
    project_id = os.environ["LS_PROJECT_ID"]

    with open("scripts/annotation/label_studio_config.xml") as f:
        schema = f.read()

    s = get_session(ls_url, email, password)
    result = upload_schema(s, ls_url, project_id, schema)
    print(f"Schema updated for project {result['id']}: {result['title']}")
