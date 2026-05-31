import os
import requests
from dotenv import load_dotenv

from scripts.annotation.pre_annotate import citations_to_prediction


def get_session(ls_url: str, email: str, password: str) -> requests.Session:
    s = requests.Session()
    s.get(ls_url + "/user/login")
    csrf = s.cookies.get("csrftoken")
    if not csrf:
        raise ValueError("CSRF token not found in cookies")
    s.post(
        ls_url + "/user/login",
        data={"email": email, "password": password, "csrfmiddlewaretoken": csrf},
        headers={"Referer": ls_url + "/user/login", "X-CSRFToken": csrf},
    )
    return s


def upload_task(s: requests.Session, ls_url: str, project_id: str, text: str) -> dict:
    prediction = citations_to_prediction(text)
    csrf = s.cookies["csrftoken"]
    if not csrf:
        raise ValueError("CSRF token not found in cookies")
    r = s.post(
        f"{ls_url}/api/projects/{project_id}/import",
        json=[{"data": {"text": text}, "predictions": [prediction]}],
        headers={"X-CSRFToken": csrf, "Referer": ls_url},
    )
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    import sys

    load_dotenv()
    ls_url     = os.environ["LS_URL"]
    email      = os.environ["LS_EMAIL"]
    password   = os.environ["LS_PASSWORD"]
    project_id = os.environ["LS_PROJECT_ID"]

    paths = sys.argv[1:]
    if not paths:
        print("Usage: python -m scripts.annotation.upload_tasks <file1.txt> [file2.txt ...]")
        sys.exit(1)

    s = get_session(ls_url, email, password)

    for path in paths:
        with open(path) as f:
            text = f.read()
        result = upload_task(s, ls_url, project_id, text)
        print(f"{path}: {result['task_count']} task, {result['prediction_count']} prediction")
