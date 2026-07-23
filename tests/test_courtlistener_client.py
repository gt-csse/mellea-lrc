"""Tests for the direct CourtListener API client."""

from __future__ import annotations

import json
import unittest
from typing import Any

import requests

from mellea_lrc.courtlistener.client import (
    CourtListenerClient,
    CourtListenerConfig,
    CourtListenerError,
)


class FakeResponse:
    """Minimal response used to test client logic without live HTTP."""

    def __init__(
        self,
        payload: Any,
        status_code: int = 200,
        url: str = "https://example.test/citation-lookup/",
        *,
        json_error: bool = False,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.url = url
        self.text = "not-json" if json_error else json.dumps(payload)
        self._json_error = json_error

    def json(self) -> Any:
        """Return the configured JSON payload or simulate invalid JSON."""
        if self._json_error:
            raise ValueError("invalid JSON")
        return self._payload


class FakeSession:
    """Record outgoing requests and return queued responses or errors."""

    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        """Return the next response after recording the request."""
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self.responses:
            raise AssertionError("No fake response queued")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def client(session: FakeSession) -> CourtListenerClient:
    """Construct a direct client with deterministic test configuration."""
    return CourtListenerClient(
        config=CourtListenerConfig(
            base_url="https://www.courtlistener.com/api/rest/v4/",
            token="token-a",
        ),
        session=session,
    )


class CourtListenerClientTests(unittest.TestCase):
    """Regression guards for exact citation lookup and transport failures."""

    def test_lookup_uses_exact_citation_post_contract(self) -> None:
        """The request contains only volume, reporter, and page."""
        session = FakeSession(
            [
                FakeResponse(
                    [
                        {
                            "citation": "347 U.S. 483",
                            "status": 200,
                            "clusters": [{"case_name": "Brown"}],
                        }
                    ]
                )
            ]
        )

        result = client(session).lookup_citation("347", "U.S.", "483")

        self.assertEqual(result.status, 200)
        self.assertEqual(result.records[0].case_name, "Brown")
        self.assertEqual(session.calls[0]["method"], "POST")
        self.assertEqual(
            session.calls[0]["url"],
            "https://www.courtlistener.com/api/rest/v4/citation-lookup/",
        )
        self.assertEqual(
            session.calls[0]["data"],
            {"volume": "347", "reporter": "U.S.", "page": "483"},
        )
        self.assertEqual(
            session.calls[0]["headers"]["User-Agent"],
            "mellea-lrc (+https://github.com/gt-csse/mellea-lrc)",
        )

    def test_empty_lookup_is_not_found(self) -> None:
        """CourtListener's explicit not-found result remains distinguishable."""
        result = client(
            FakeSession([FakeResponse([{"citation": "1 U.S. 9999", "status": 404, "clusters": []}])])
        ).lookup_citation("1", "U.S.", "9999")

        self.assertEqual(result.status, 404)
        self.assertEqual(result.citation, "1 U.S. 9999")
        self.assertEqual(result.records, ())

    def test_get_docket_returns_court_id(self) -> None:
        """The docket endpoint exposes the authoritative CourtListener court ID."""
        session = FakeSession(
            [FakeResponse({"id": 123, "court": "https://www.courtlistener.com/api/rest/v4/courts/ca2/"})]
        )

        result = client(session).get_docket("123")

        self.assertEqual(result.docket_id, "123")
        self.assertEqual(result.court_id, "ca2")
        self.assertEqual(session.calls[0]["method"], "GET")
        self.assertEqual(session.calls[0]["url"], "https://www.courtlistener.com/api/rest/v4/dockets/123/")

    def test_ambiguous_lookup_preserves_each_candidate(self) -> None:
        """A 300 response retains every candidate returned for the locator."""
        result = client(
            FakeSession(
                [
                    FakeResponse(
                        [
                            {
                                "citation": "1 F.2d 2",
                                "status": 300,
                                "clusters": [
                                    {"caseName": "First", "docketId": 10},
                                    {"caseName": "Second", "docketId": 20},
                                ],
                            }
                        ]
                    )
                ]
            )
        ).lookup_citation("1", "F.2d", "2")

        self.assertEqual(result.status, 300)
        self.assertEqual([record.case_name for record in result.records], ["First", "Second"])
        self.assertEqual([record.docket_id for record in result.records], ["10", "20"])

    def test_rate_limit_is_typed_after_one_request(self) -> None:
        """A rate-limit response raises an API-limit error after one request."""
        session = FakeSession([FakeResponse({"detail": "rate limited"}, status_code=429)])

        with self.assertRaises(CourtListenerError) as raised:
            client(session).lookup_citation("1", "U.S.", "1")

        error = raised.exception
        self.assertEqual(error.failure_type, "api_limit")
        self.assertEqual(error.upstream_status_code, 429)
        self.assertIs(error.retryable, True)
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(session.calls[0]["headers"]["Authorization"], "Token token-a")

    def test_search_uses_get_with_supported_type_and_pagination(self) -> None:
        """Search forwards the documented query parameters and exposes cursors."""
        session = FakeSession(
            [
                FakeResponse(
                    {
                        "count": 1,
                        "next": "https://example.test/search/?cursor=next-page",
                        "previous": None,
                        "results": [{"id": 42, "caseName": "Brown"}],
                    },
                    url="https://example.test/search/",
                )
            ]
        )

        result = client(session).search("Brown", "o", cursor="current-page", semantic=True)

        self.assertEqual(result.query, "Brown")
        self.assertEqual(result.search_type, "o")
        self.assertIs(result.semantic, True)
        self.assertEqual(result.count, 1)
        self.assertEqual(result.results[0]["caseName"], "Brown")
        self.assertEqual(result.next_cursor, "next-page")
        self.assertIsNone(result.previous_cursor)
        self.assertEqual(session.calls[0]["method"], "GET")
        self.assertEqual(session.calls[0]["url"], "https://www.courtlistener.com/api/rest/v4/search/")
        self.assertEqual(
            session.calls[0]["params"],
            {"q": "Brown", "type": "o", "cursor": "current-page", "semantic": "true"},
        )

    def test_search_rejects_unsupported_type(self) -> None:
        """The public wrapper exposes only the CourtListener v4 search corpora."""
        with self.assertRaises(ValueError):
            client(FakeSession([])).search("Brown", "p")  # type: ignore[arg-type]

    def test_search_invalid_response_is_typed(self) -> None:
        """A malformed search response is rejected at the client boundary."""
        with self.assertRaises(CourtListenerError) as raised:
            client(FakeSession([FakeResponse({"results": []})])).search("Brown", "o")

        self.assertEqual(raised.exception.failure_type, "upstream_invalid_response")

    def test_auth_failure_is_typed(self) -> None:
        """CourtListener authentication failures remain distinguishable."""
        session = FakeSession([FakeResponse({"detail": "bad token"}, status_code=403)])

        with self.assertRaises(CourtListenerError) as raised:
            client(session).lookup_citation("1", "U.S.", "1")

        self.assertEqual(raised.exception.failure_type, "upstream_auth")
        self.assertEqual(raised.exception.upstream_status_code, 403)
        self.assertIs(raised.exception.retryable, False)

    def test_invalid_json_is_typed(self) -> None:
        """A successful non-JSON response is rejected at the HTTP boundary."""
        session = FakeSession([FakeResponse(None, json_error=True)])

        with self.assertRaises(CourtListenerError) as raised:
            client(session).lookup_citation("1", "U.S.", "1")

        self.assertEqual(raised.exception.failure_type, "upstream_invalid_json")
        self.assertEqual(raised.exception.upstream_detail, "not-json")

    def test_timeout_is_typed(self) -> None:
        """A transport timeout exposes a retryable timeout failure."""
        session = FakeSession([requests.Timeout("slow")])

        with self.assertRaises(CourtListenerError) as raised:
            client(session).lookup_citation("1", "U.S.", "1")

        self.assertEqual(raised.exception.failure_type, "upstream_timeout")
        self.assertIsNone(raised.exception.upstream_status_code)
        self.assertIs(raised.exception.retryable, True)

    def test_invalid_response_shape_is_typed(self) -> None:
        """A response violating the one-result contract becomes a client error."""
        session = FakeSession([FakeResponse([])])

        with self.assertRaises(CourtListenerError) as raised:
            client(session).lookup_citation("1", "U.S.", "1")

        self.assertEqual(raised.exception.failure_type, "upstream_invalid_response")
        self.assertEqual(raised.exception.upstream_status_code, 200)
        self.assertIs(raised.exception.retryable, False)


if __name__ == "__main__":
    unittest.main()
