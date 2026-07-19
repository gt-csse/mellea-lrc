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
from mellea_lrc.courtlistener.citation_lookup_models import (
    CourtListenerCitationLookup,
    CourtListenerCitationRecord,
)
from mellea_lrc.courtlistener.search_models import CourtListenerSearchResult


class FailingClient:
    def get_docket(self, _: int) -> dict:
        raise CourtListenerError(
            "CourtListener GET failed with 429",
            failure_type="api_limit",
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
            records=(CourtListenerCitationRecord(case_name="Brown"),),
            cache="miss",
        )


class ClusterClient:
    def get_cluster(self, cl_cluster_id: int) -> dict[str, object]:
        return {"cl_cluster_id": cl_cluster_id, "docket_id": 5068645}


class SearchClient:
    def __init__(self) -> None:
        self.semantic: bool | None = None

    def search(
        self,
        *,
        q: str,
        search_type: str,
        cursor: str | None,
        semantic: bool,
    ) -> CourtListenerSearchResult:
        self.semantic = semantic
        return CourtListenerSearchResult(
            query=q,
            search_type=search_type,
            semantic=semantic,
            http_status=200,
            count=0,
            records=(),
            next_cursor=cursor,
        )


class AppTests(unittest.TestCase):
    def test_search_route_forwards_semantic_mode(self) -> None:
        search_client = SearchClient()

        response = TestClient(create_api(client_factory=lambda: search_client)).get(
            "/search",
            params={"q": "Johnson v. City of Shelby", "type": "o", "semantic": "true"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIs(search_client.semantic, True)
        self.assertIs(response.json()["semantic"], True)

    def test_search_route_defaults_to_keyword_mode(self) -> None:
        search_client = SearchClient()

        response = TestClient(create_api(client_factory=lambda: search_client)).get(
            "/search",
            params={"q": "Johnson v. City of Shelby", "type": "o"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIs(search_client.semantic, False)

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
        self.assertEqual(response.json()["status"], 200)
        self.assertEqual(response.json()["citation"], "347 U.S. 483")

    def test_citation_lookup_route_requires_locator_parts(self) -> None:
        response = TestClient(create_api(client_factory=CitationClient)).post(
            "/citation-lookup",
            data={"volume": "347", "reporter": "U.S."},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "page is required"})

    def test_cluster_route_returns_linked_docket_id(self) -> None:
        response = TestClient(create_api(client_factory=ClusterClient)).get("/clusters/1969711")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"cl_cluster_id": 1969711, "docket_id": 5068645})

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
