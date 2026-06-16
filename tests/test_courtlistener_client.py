from __future__ import annotations

import json
import unittest
from typing import Any

from mellea_lrc.courtlistener.client import (
    CourtListenerClient,
    CourtListenerConfig,
    CourtListenerError,
    CourtListenerRateLimitConfig,
    CourtListenerRateLimiter,
)


class FakeResponse:
    """Minimal response object used to test client logic without live HTTP."""

    def __init__(self, payload: Any, status_code: int = 200, url: str = "https://example.test") -> None:
        self._payload = payload
        self.status_code = status_code
        self.url = url
        self.text = json.dumps(payload)

    def json(self) -> Any:
        return self._payload


class FakeSession:
    """Records outgoing requests and returns queued fake responses."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self.responses:
            raise AssertionError("No fake response queued")
        return self.responses.pop(0)


def client(
    session: FakeSession,
    rate_limiter: CourtListenerRateLimiter | None = None,
) -> CourtListenerClient:
    return CourtListenerClient(
        config=CourtListenerConfig(
            base_url="https://www.courtlistener.com/api/rest/v4/",
            tokens=("token-a", "token-b"),
        ),
        session=session,
        rate_limiter=rate_limiter,
    )


class CourtListenerClientTests(unittest.TestCase):
    """Regression guards for request shaping, response normalization, and token use."""

    def test_get_returns_cache_envelope(self) -> None:
        """GET helpers should return our stable cache envelope and browser-like headers."""
        session = FakeSession([FakeResponse({"id": 1, "court_id": "dcd"})])

        result = client(session).get_docket(1)

        self.assertEqual(result["cache"], "miss")
        self.assertEqual(result["cl_docket_id"], 1)
        self.assertEqual(result["court_id"], "dcd")
        self.assertEqual(result["raw"], {"id": 1, "court_id": "dcd"})
        self.assertEqual(session.calls[0]["method"], "GET")
        self.assertEqual(session.calls[0]["url"], "https://www.courtlistener.com/api/rest/v4/dockets/1/")
        self.assertIn("Mozilla/5.0", session.calls[0]["headers"]["User-Agent"])

    def test_resolve_docket_uses_court_and_docket_number(self) -> None:
        """Docket resolution should use CourtListener's official docket filters."""
        session = FakeSession(
            [
                FakeResponse(
                    {
                        "count": 1,
                        "results": [
                            {
                                "id": 4214664,
                                "court_id": "dcd",
                                "docket_number": "1:16-cv-00745",
                                "case_name": "Example",
                            }
                        ],
                    }
                )
            ]
        )

        result = client(session).resolve_docket("dcd", "1:16-cv-00745")

        self.assertEqual(result["candidates"][0]["cl_docket_id"], 4214664)
        self.assertEqual(result["raw"]["count"], 1)
        self.assertEqual(
            session.calls[0]["params"],
            {"court": "dcd", "docket_number": "1:16-cv-00745"},
        )

    def test_docket_entry_search_includes_nested_recap_documents(self) -> None:
        """Entry search should expose nested RECAP documents and one availability meaning."""
        session = FakeSession(
            [
                FakeResponse(
                    {
                        "count": 1,
                        "results": [
                            {
                                "id": 77,
                                "docket": "https://www.courtlistener.com/api/rest/v4/dockets/4214664/",
                                "entry_number": 4,
                                "recap_documents": [
                                    {
                                        "id": 88,
                                        "docket_entry": (
                                            "https://www.courtlistener.com/api/rest/v4/docket-entries/77/"
                                        ),
                                        "document_number": 4,
                                        "filepath_local": "",
                                        "filepath_ia": "https://archive.org/example.pdf",
                                    }
                                ],
                            }
                        ],
                    }
                )
            ]
        )

        result = client(session).search_docket_entries(4214664, 4)

        document = result["entries"][0]["recap_documents"][0]
        self.assertEqual(result["cl_docket_id"], 4214664)
        self.assertEqual(result["entry_number"], "4")
        self.assertEqual(document["recap_document_id"], 88)
        self.assertIs(document["available"], True)
        self.assertNotIn("local_available", document)
        self.assertEqual(
            session.calls[0]["params"],
            {"docket": 4214664, "entry_number": 4},
        )

    def test_docket_entry_search_can_list_entries_without_entry_number(self) -> None:
        """Entry listing should work when the LLM has not identified an exact ECF."""
        session = FakeSession(
            [
                FakeResponse(
                    {
                        "count": 1,
                        "results": [
                            {
                                "id": 77,
                                "docket": "https://www.courtlistener.com/api/rest/v4/dockets/4214664/",
                                "entry_number": 4,
                                "description": "Motion for summary judgment",
                            }
                        ],
                    }
                )
            ]
        )

        result = client(session).search_docket_entries(4214664)

        self.assertEqual(result["cl_docket_id"], 4214664)
        self.assertNotIn("entry_number", result)
        self.assertEqual(result["entries"][0]["entry_number"], "4")
        self.assertEqual(session.calls[0]["params"], {"docket": 4214664})
        self.assertNotIn("cursor", session.calls[0]["params"])

    def test_docket_entry_search_forwards_cursor_with_docket_filter(self) -> None:
        """Cursor pagination should keep CourtListener's base docket query intact."""
        session = FakeSession(
            [
                FakeResponse(
                    {
                        "count": 2,
                        "next": (
                            "https://www.courtlistener.com/api/rest/v4/docket-entries/"
                            "?cursor=next%2Fpage%3D&docket=4214664"
                        ),
                        "previous": (
                            "https://www.courtlistener.com/api/rest/v4/docket-entries/"
                            "?cursor=previous%2Fpage%3D&docket=4214664"
                        ),
                        "results": [
                            {
                                "id": 77,
                                "docket": "https://www.courtlistener.com/api/rest/v4/dockets/4214664/",
                                "entry_number": 4,
                            }
                        ],
                    }
                )
            ]
        )

        result = client(session).search_docket_entries(
            4214664,
            entry_number=4,
            cursor="current/page=",
        )

        self.assertEqual(
            session.calls[0]["params"],
            {"docket": 4214664, "entry_number": 4, "cursor": "current/page="},
        )
        self.assertEqual(result["next_cursor"], "next/page=")
        self.assertEqual(result["previous_cursor"], "previous/page=")

    def test_docket_entry_search_omits_missing_cursor(self) -> None:
        """A missing cursor should not send an empty cursor filter upstream."""
        session = FakeSession(
            [
                FakeResponse(
                    {
                        "count": 0,
                        "next": None,
                        "previous": None,
                        "results": [],
                    }
                )
            ]
        )

        result = client(session).search_docket_entries(4214664, cursor=None)

        self.assertEqual(session.calls[0]["params"], {"docket": 4214664})
        self.assertIsNone(result["next_cursor"])
        self.assertIsNone(result["previous_cursor"])

    def test_recap_document_search_by_entry_id(self) -> None:
        """RECAP document search should expose CourtListener's docket_entry filter."""
        session = FakeSession(
            [
                FakeResponse(
                    {
                        "count": 1,
                        "results": [
                            {
                                "id": 88,
                                "docket_entry": (
                                    "https://www.courtlistener.com/api/rest/v4/docket-entries/77/"
                                ),
                                "document_number": 4,
                                "filepath_local": "recap/example.pdf",
                                "filepath_ia": "",
                            }
                        ],
                    }
                )
            ]
        )

        result = client(session).search_recap_documents(cl_docket_entry_id=77)

        self.assertEqual(result["cl_docket_entry_id"], 77)
        self.assertEqual(result["documents"][0]["recap_document_id"], 88)
        self.assertIs(result["documents"][0]["available"], True)
        self.assertEqual(session.calls[0]["params"], {"docket_entry": 77})

    def test_recap_document_search_by_docket_and_entry_reuses_entry_documents(self) -> None:
        """The workflow can resolve documents from cl_docket_id plus ECF."""
        session = FakeSession(
            [
                FakeResponse(
                    {
                        "count": 1,
                        "results": [
                            {
                                "id": 77,
                                "docket": "https://www.courtlistener.com/api/rest/v4/dockets/4214664/",
                                "entry_number": 4,
                                "recap_documents": [
                                    {
                                        "id": 88,
                                        "document_number": 4,
                                        "filepath_local": "recap/example.pdf",
                                        "filepath_ia": "",
                                    }
                                ],
                            }
                        ],
                    }
                )
            ]
        )

        result = client(session).search_recap_documents(cl_docket_id=4214664, entry_number=4)

        self.assertEqual(result["cl_docket_id"], 4214664)
        self.assertEqual(result["entry_number"], "4")
        self.assertEqual(result["documents"][0]["recap_document_id"], 88)
        self.assertEqual(
            session.calls[0]["params"],
            {"docket": 4214664, "entry_number": 4},
        )

    def test_get_recap_document_normalizes_availability(self) -> None:
        """A RECAP document is available iff filepath_local or filepath_ia is present."""
        session = FakeSession(
            [
                FakeResponse(
                    {
                        "id": 88,
                        "document_number": "4",
                        "filepath_local": None,
                        "filepath_ia": "",
                    }
                )
            ]
        )

        result = client(session).get_recap_document(88)

        self.assertEqual(result["recap_document_id"], 88)
        self.assertIs(result["available"], False)
        self.assertEqual(result["raw"]["id"], 88)

    def test_download_url_prefers_courtlistener_storage(self) -> None:
        """Available local storage paths should normalize to a usable download URL."""
        session = FakeSession(
            [
                FakeResponse(
                    {
                        "id": 88,
                        "filepath_local": "recap/example.pdf",
                        "filepath_ia": "",
                    }
                )
            ]
        )

        result = client(session).get_recap_document_download_url(88)

        self.assertIs(result["available"], True)
        self.assertEqual(result["source"], "courtlistener_storage")
        self.assertEqual(result["download_url"], "https://storage.courtlistener.com/recap/example.pdf")

    def test_get_court_normalizes_court_metadata(self) -> None:
        """Court metadata should be available for court-id validation."""
        session = FakeSession(
            [
                FakeResponse(
                    {
                        "id": "dcd",
                        "full_name": "District Court, District of Columbia",
                        "citation_string": "D.D.C.",
                        "jurisdiction": "FD",
                    }
                )
            ]
        )

        result = client(session).get_court("dcd")

        self.assertEqual(result["court_id"], "dcd")
        self.assertEqual(result["citation_string"], "D.D.C.")
        self.assertEqual(session.calls[0]["url"], "https://www.courtlistener.com/api/rest/v4/courts/dcd/")

    def test_search_supports_official_v4_types(self) -> None:
        """Search should preserve raw results and normalize candidate fields per type."""
        session = FakeSession(
            [
                FakeResponse(
                    {
                        "count": 1,
                        "results": [
                            {
                                "docket_id": 4214664,
                                "court_id": "dcd",
                                "docketNumber": "1:16-cv-00745",
                                "caseName": "Example",
                                "more_docs": False,
                                "recap_documents": [
                                    {
                                        "id": 88,
                                        "filepath_local": "/tmp/example.pdf",
                                        "filepath_ia": "",
                                    }
                                ],
                            }
                        ],
                    }
                )
            ]
        )

        result = client(session).search("Example", "r")

        self.assertEqual(result["type"], "r")
        self.assertEqual(result["results"][0]["cl_docket_id"], 4214664)
        self.assertEqual(result["results"][0]["recap_documents"][0]["recap_document_id"], 88)
        self.assertIs(result["results"][0]["recap_documents"][0]["available"], True)
        self.assertEqual(session.calls[0]["params"], {"q": "Example", "type": "r"})

    def test_search_rejects_unlisted_types(self) -> None:
        """The public search wrapper should not silently accept unsupported types."""
        with self.assertRaises(ValueError):
            client(FakeSession([])).search("Example", "p")

    def test_citation_lookup_uses_post(self) -> None:
        """Citation lookup must preserve CourtListener's volume/reporter/page POST shape."""
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

        self.assertEqual(result.cache, "miss")
        self.assertEqual(result.status, 200)
        self.assertEqual(result.clusters, ({"case_name": "Brown"},))
        self.assertEqual(session.calls[0]["method"], "POST")
        self.assertEqual(
            session.calls[0]["data"],
            {"volume": "347", "reporter": "U.S.", "page": "483"},
        )

    def test_empty_citation_lookup_is_not_found(self) -> None:
        """An empty CourtListener citation response should normalize to our 404 object."""
        result = client(FakeSession([FakeResponse([])])).lookup_citation("1", "U.S.", "9999")

        self.assertEqual(result.cache, "miss")
        self.assertEqual(result.status, 404)
        self.assertEqual(result.citation, "1 U.S. 9999")

    def test_rate_limit_rotates_token(self) -> None:
        """A 429 should retry with the next configured API token."""
        session = FakeSession(
            [
                FakeResponse({"detail": "rate limited"}, status_code=429),
                FakeResponse({"id": 9}),
            ]
        )

        self.assertEqual(client(session).get_docket(9)["raw"], {"id": 9})
        self.assertEqual(session.calls[0]["headers"]["Authorization"], "Token token-a")
        self.assertEqual(session.calls[1]["headers"]["Authorization"], "Token token-b")

    def test_exhausted_rate_limit_surfaces_api_limit_failure(self) -> None:
        """If every token is rate limited, callers should see the explicit API-limit type."""
        session = FakeSession(
            [
                FakeResponse({"detail": "rate limited"}, status_code=429),
                FakeResponse({"detail": "rate limited"}, status_code=429),
            ]
        )

        with self.assertRaises(CourtListenerError) as raised:
            client(session).get_docket(9)

        error = raised.exception
        self.assertEqual(error.failure_type, "api_limit")
        self.assertEqual(error.status_code, 429)
        self.assertEqual(error.upstream_status_code, 429)
        self.assertIs(error.retryable, True)
        self.assertEqual(error.upstream_detail, {"detail": "rate limited"})
        self.assertEqual(error.to_public_dict()["failure_type"], "api_limit")

    def test_upstream_forbidden_surfaces_auth_failure(self) -> None:
        """CourtListener auth failures should be distinguishable from rate limits."""
        session = FakeSession([FakeResponse({"detail": "bad token"}, status_code=403)])

        with self.assertRaises(CourtListenerError) as raised:
            client(session).get_docket(9)

        error = raised.exception
        self.assertEqual(error.failure_type, "upstream_auth")
        self.assertEqual(error.status_code, 502)
        self.assertEqual(error.upstream_status_code, 403)
        self.assertIs(error.retryable, False)

    def test_rate_limiter_waits_before_exceeding_window(self) -> None:
        """The client should wait locally instead of sending over-window requests."""
        now = 0.0
        sleeps: list[float] = []

        def clock() -> float:
            return now

        def sleeper(seconds: float) -> None:
            nonlocal now
            sleeps.append(seconds)
            now += seconds

        limiter = CourtListenerRateLimiter(
            CourtListenerRateLimitConfig(
                per_minute=2,
                per_hour=0,
                per_day=0,
                max_wait_seconds=60,
            ),
            clock=clock,
            sleeper=sleeper,
        )
        session = FakeSession(
            [
                FakeResponse({"id": 1}),
                FakeResponse({"id": 2}),
                FakeResponse({"id": 3}),
            ]
        )
        api = client(session, rate_limiter=limiter)

        api.get_docket(1)
        api.get_docket(2)
        api.get_docket(3)

        self.assertEqual(sleeps, [60.0])
        self.assertEqual(len(session.calls), 3)

    def test_rate_limiter_surfaces_api_limit_when_wait_exceeds_budget(self) -> None:
        """Long local waits should return the same typed API-limit failure."""
        limiter = CourtListenerRateLimiter(
            CourtListenerRateLimitConfig(
                per_minute=1,
                per_hour=0,
                per_day=0,
                max_wait_seconds=5,
            ),
            clock=lambda: 0.0,
            sleeper=lambda _: None,
        )
        session = FakeSession([FakeResponse({"id": 1}), FakeResponse({"id": 2})])
        api = client(session, rate_limiter=limiter)

        api.get_docket(1)
        with self.assertRaises(CourtListenerError) as raised:
            api.get_docket(2)

        error = raised.exception
        self.assertEqual(error.failure_type, "api_limit")
        self.assertEqual(error.status_code, 429)
        self.assertEqual(error.retry_after_seconds, 60.0)
        self.assertEqual(len(session.calls), 1)


if __name__ == "__main__":
    unittest.main()
