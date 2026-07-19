"""Tests for the CourtListener citation lookup boundary."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from mellea_lrc.courtlistener.citation_lookup import normalize_citation_lookup_payload


class CourtListenerLookupTests(unittest.TestCase):
    """Guard strict known-field validation without retaining unknown JSON."""

    def test_unknown_fields_are_ignored(self) -> None:
        lookup = normalize_citation_lookup_payload(
            [
                {
                    "citation": "347 U.S. 483",
                    "status": 200,
                    "unknown_response_field": "ignored",
                    "clusters": [
                        {
                            "caseName": "Brown v. Board of Education",
                            "dateFiled": "1954-05-17",
                            "docketId": 123,
                            "unknown_cluster_field": "ignored",
                        }
                    ],
                }
            ]
        )

        self.assertEqual(lookup.records[0].case_name, "Brown v. Board of Education")
        self.assertEqual(lookup.records[0].docket_id, "123")
        self.assertFalse(hasattr(lookup, "extra_data"))
        self.assertFalse(hasattr(lookup.records[0], "extra_data"))

    def test_invalid_known_field_type_is_rejected(self) -> None:
        payload = [{"citation": "347 U.S. 483", "status": "200", "clusters": []}]

        with self.assertRaises(ValidationError):
            normalize_citation_lookup_payload(payload)

    def test_explicit_not_found_result_is_preserved(self) -> None:
        lookup = normalize_citation_lookup_payload(
            [{"citation": "1 U.S. 9999", "status": 404, "clusters": []}]
        )

        self.assertEqual(lookup.citation, "1 U.S. 9999")
        self.assertEqual(lookup.status, 404)
        self.assertEqual(lookup.records, ())

    def test_response_requires_exactly_one_result(self) -> None:
        for payload in (
            [],
            [
                {"citation": "1 U.S. 1", "status": 200},
                {"citation": "2 U.S. 2", "status": 200},
            ],
        ):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                normalize_citation_lookup_payload(payload)

    def test_response_rejects_an_unwrapped_result(self) -> None:
        with self.assertRaises(ValidationError):
            normalize_citation_lookup_payload({"citation": "1 U.S. 1", "status": 200})

    def test_result_requires_citation_and_status(self) -> None:
        for payload in ([{"status": 200}], [{"citation": "1 U.S. 1"}]):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                normalize_citation_lookup_payload(payload)


if __name__ == "__main__":
    unittest.main()
