"""Upload citations and text to Label Studio."""

import os
import requests
from dotenv import load_dotenv


def _get_session(ls_url: str, email: str, password: str) -> requests.Session:
    """Return a cookie percistance session."""
    session = requests.Session()
    session.get(ls_url + "/user/login")
    csrf = session.cookies.get("csrftoken")
    if not csrf:
        message = "CSRF toek not found in cookies"
        raise ValueError(message)
    session.post(
        ls_url + "/user/login",
        data={"email": email, "password": password, "csrfmiddlewaretoken": csrf},
        headers={"Referer": ls_url + "/user/login", "X-CSRFToken": csrf},
    )
    return session


def _upload_task(
    session: requests.Session, ls_url: str, project_id: str, text: str, predictions: dict
) -> dict:
    """Upload data to Lable Studio."""
    csrf = session.cookies["csrftoken"]
    if not csrf:
        message = "CSRF toek not found in cookies"
        raise ValueError(message)
    r = session.post(
        f"{ls_url}/api/projects/{project_id}/import",
        json=[{"data": {"text": text}, "predictions": [predictions]}],
        headers={"X-CSRFToken": csrf, "Referer": ls_url},
    )
    r.raise_for_status()
    return r.json()


def send_data(text: str, predictions: dict) -> None:
    """Upload citations to label studio."""
    load_dotenv()
    ls_url = os.environ["LS_URL"]
    email = os.environ["LS_EMAIL"]
    password = os.environ["LS_PASSWORD"]
    project_id = os.environ["LS_PROJECT_ID"]
    session = _get_session(ls_url, email, password)
    _upload_task(session, ls_url, project_id, text, predictions)
