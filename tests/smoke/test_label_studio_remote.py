"""Remote smoke tests for the deployed Label Studio Modal backend."""

from __future__ import annotations

import requests


def test_label_studio_health(label_studio_url: str, remote_timeout: float) -> None:
    response = requests.get(f"{label_studio_url}/health", timeout=remote_timeout)

    assert response.status_code == 200
    assert response.json() == {"status": "UP", "model_class": "mellea-lrc-prototype"}


def test_label_studio_predict_text_without_validation(
    label_studio_url: str,
    remote_timeout: float,
) -> None:
    response = requests.post(
        f"{label_studio_url}/predict_text",
        json={"text": "Brown v. Board, 347 U.S. 483.", "validate": False},
        timeout=remote_timeout,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "Brown v. Board, 347 U.S. 483."
    assert payload["validation"] is None
    assert payload["stats"]["citation_spans"] >= 1
    assert payload["prediction"]["result"]
