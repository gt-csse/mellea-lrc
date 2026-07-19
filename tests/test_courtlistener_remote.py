from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from mellea_lrc.courtlistener.remote import (
    CourtListenerAccessClient,
    CourtListenerAccessConfig,
)


class CourtListenerAccessClientTests(unittest.TestCase):
    """Contract tests for an already-deployed CourtListener access service."""

    def test_config_reads_access_url(self) -> None:
        with patch.dict(os.environ, {"CL_ACCESS_URL": "https://cl.example.test/"}):
            config = CourtListenerAccessConfig.from_env()

        self.assertEqual(config.base_url, "https://cl.example.test/")

    def test_config_requires_access_url(self) -> None:
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            CourtListenerAccessConfig.from_env()

    def test_lookup_citation_posts_service_contract_and_normalizes_result(self) -> None:
        calls: list[tuple[str, dict[str, str]]] = []

        def post_json(url: str, data: dict[str, str]) -> object:
            calls.append((url, data))
            return {
                "cache": "hit",
                "key": "citation-key",
                "response": {
                    "citation": "347 U.S. 483",
                    "status": 200,
                    "clusters": [
                        {
                            "case_name": "Brown v. Board of Education",
                            "date_filed": "1954-05-17",
                            "docket_id": 123,
                        }
                    ],
                },
            }

        client = CourtListenerAccessClient(
            CourtListenerAccessConfig("https://cl.example.test/"),
            post_json=post_json,
        )

        result = client.lookup_citation("347", "U.S.", "483")

        self.assertEqual(
            calls,
            [
                (
                    "https://cl.example.test/citation-lookup",
                    {"volume": "347", "reporter": "U.S.", "page": "483"},
                )
            ],
        )
        self.assertEqual(result.status, 200)
        self.assertEqual(result.cache, "hit")
        self.assertEqual(result.records[0].case_name, "Brown v. Board of Education")
        self.assertEqual(result.records[0].docket_id, "123")

    def test_get_docket_builds_route_and_marks_success(self) -> None:
        calls: list[str] = []

        def get_json(url: str) -> object:
            calls.append(url)
            return {"cl_docket_id": 42, "case_name": "Example"}

        client = CourtListenerAccessClient(
            CourtListenerAccessConfig("https://cl.example.test"),
            get_json=get_json,
        )

        result = client.get_docket(42)

        self.assertEqual(calls, ["https://cl.example.test/dockets/42"])
        self.assertEqual(result["http_status"], 200)

    def test_search_encodes_query_and_preserves_service_failure(self) -> None:
        calls: list[str] = []

        def get_json(url: str) -> object:
            calls.append(url)
            return {"http_status": 429, "detail": "rate limited"}

        client = CourtListenerAccessClient(
            CourtListenerAccessConfig("https://cl.example.test"),
            get_json=get_json,
        )

        result = client.search_opinions("Smith & Jones")

        self.assertEqual(
            calls,
            ["https://cl.example.test/search?q=Smith+%26+Jones&type=o"],
        )
        self.assertEqual(result, {"http_status": 429, "detail": "rate limited"})

    def test_non_http_service_url_is_rejected_before_transport(self) -> None:
        client = CourtListenerAccessClient(
            CourtListenerAccessConfig("file:///tmp/service"),
            get_json=lambda _: {},
        )

        with self.assertRaises(ValueError):
            client.get_docket(1)


if __name__ == "__main__":
    unittest.main()
