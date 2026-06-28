from __future__ import annotations

import unittest
import warnings

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated",
)

from fastapi.testclient import TestClient

from mellea_lrc.courtlistener.api import create_api
from mellea_lrc.courtlistener.client import CourtListenerError
from mellea_lrc.courtlistener.types import CitationMatch, CourtListenerCitationLookup


class FailingClient:
    def get_docket(self, _: int) -> dict:
        raise CourtListenerError(
            "CourtListener GET failed with 429",
            failure_type="api_limit",
            status_code=429,
            upstream_status_code=429,
            retryable=True,
            cache_key="cache-key",
            upstream_detail={"detail": "rate limited"},
        )


class CitationClient:
    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        return CourtListenerCitationLookup(
            citation=f"{volume} {reporter} {page}",
            status=200,
            matches=(CitationMatch(case_name="Brown"),),
            cache="miss",
        )


class AppTests(unittest.TestCase):
    def test_courtlistener_error_handler_returns_typed_failure(self) -> None:
        response = TestClient(create_api(client_factory=FailingClient)).get("/dockets/9")

        self.assertEqual(response.status_code, 429)
        self.assertEqual(
            response.json(),
            {
                "detail": {
                    "failure_type": "api_limit",
                    "message": "CourtListener GET failed with 429",
                    "retryable": True,
                    "upstream_status_code": 429,
                    "key": "cache-key",
                    "upstream_detail": {"detail": "rate limited"},
                }
            },
        )

    def test_citation_lookup_route_uses_form_contract(self) -> None:
        response = TestClient(create_api(client_factory=CitationClient)).post(
            "/citation-lookup",
            data={"volume": "347", "reporter": "U.S.", "page": "483"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"]["status"], 200)
        self.assertEqual(response.json()["response"]["citation"], "347 U.S. 483")

    def test_citation_lookup_route_requires_locator_parts(self) -> None:
        response = TestClient(create_api(client_factory=CitationClient)).post(
            "/citation-lookup",
            data={"volume": "347", "reporter": "U.S."},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "page is required"})

    def test_unknown_route_returns_bad_route_failure(self) -> None:
        response = TestClient(create_api(client_factory=CitationClient)).get("/missing-route")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json(),
            {
                "detail": {
                    "failure_type": "bad_route",
                    "message": "No matching backend route",
                    "retryable": False,
                    "method": "GET",
                    "path": "/missing-route",
                }
            },
        )

    def test_wrong_method_returns_bad_route_failure(self) -> None:
        response = TestClient(create_api(client_factory=CitationClient)).get("/citation-lookup")

        self.assertEqual(response.status_code, 405)
        self.assertEqual(
            response.json(),
            {
                "detail": {
                    "failure_type": "bad_route",
                    "message": "HTTP method is not allowed for this backend route",
                    "retryable": False,
                    "method": "GET",
                    "path": "/citation-lookup",
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
