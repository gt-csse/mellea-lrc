"""Tests for the shared CourtListener citation lookup boundary."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from mellea_lrc.courtlistener.lookup import (
    citation_lookup_envelope_dict,
    normalize_citation_lookup_payload,
)


class CourtListenerLookupTests(unittest.TestCase):
    """Guard strict validation, normalization, and lossless unknown fields."""

    def test_unknown_fields_survive_domain_round_trip(self) -> None:
        payload = {
            "cache": "miss",
            "trace_id": "request-1",
            "response": {
                "citation": "347 U.S. 483",
                "status": 200,
                "source": "courtlistener",
                "clusters": [
                    {
                        "caseName": "Brown v. Board of Education",
                        "dateFiled": "1954-05-17",
                        "docketId": 123,
                        "absolute_url": "/opinion/105221/brown-v-board-of-education/",
                    }
                ],
            },
        }

        lookup = normalize_citation_lookup_payload(payload, "347", "U.S.", "483")
        serialized = citation_lookup_envelope_dict(lookup)

        self.assertEqual(lookup.records[0].docket_id, "123")
        self.assertEqual(
            lookup.records[0].extra_data.to_dict(),
            {"absolute_url": "/opinion/105221/brown-v-board-of-education/"},
        )
        self.assertEqual(serialized["extra_data"], {"trace_id": "request-1"})
        self.assertEqual(serialized["response"]["extra_data"], {"source": "courtlistener"})

    def test_invalid_external_types_raise_at_boundary(self) -> None:
        payload = {
            "response": {
                "citation": "347 U.S. 483",
                "status": "200",
                "clusters": [],
            }
        }

        with self.assertRaises(ValidationError):
            normalize_citation_lookup_payload(payload, "347", "U.S.", "483")

    def test_empty_payload_becomes_explicit_not_found(self) -> None:
        lookup = normalize_citation_lookup_payload({}, "1", "U.S.", "9999")

        self.assertEqual(lookup.citation, "1 U.S. 9999")
        self.assertEqual(lookup.status, 404)
        self.assertEqual(lookup.records, ())


if __name__ == "__main__":
    unittest.main()
