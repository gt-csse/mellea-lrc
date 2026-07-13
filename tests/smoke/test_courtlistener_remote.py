"""Remote smoke tests for a deployed CourtListener access service."""

from __future__ import annotations

import requests
import pytest

pytestmark = pytest.mark.remote_smoke


def test_courtlistener_health(courtlistener_url: str, remote_timeout: float) -> None:
    response = requests.get(f"{courtlistener_url}/health", timeout=remote_timeout)

    assert response.status_code == 200
    assert response.json() == {"status": "UP", "service": "courtlistener-access"}


def test_courtlistener_citation_lookup(courtlistener_url: str, remote_timeout: float) -> None:
    response = requests.post(
        f"{courtlistener_url}/citation-lookup",
        data={"volume": "347", "reporter": "U.S.", "page": "483"},
        timeout=remote_timeout,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response"]["citation"] == "347 U.S. 483"
    assert payload["response"]["status"] in {200, 300}
    assert isinstance(payload["response"]["clusters"], list)
    assert payload["cache"] in {"hit", "miss"}
