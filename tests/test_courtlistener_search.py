"""Tests for the CourtListener Opinion and RECAP search boundary."""

# ruff: noqa: INP001

import pytest
from pydantic import ValidationError

from mellea_lrc.courtlistener.search import normalize_search_payload, search_result_dict


def test_opinion_search_normalizes_aliases_and_pagination() -> None:
    """Opinion results normalize aliases, IDs, cache metadata, and cursors."""
    result = normalize_search_payload(
        {
            "status": 200,
            "cache": "miss",
            "key": "search-key",
            "response": {
                "count": 1,
                "next": "https://www.courtlistener.com/api/rest/v4/search/?cursor=next-page",
                "results": [
                    {
                        "id": 1969711,
                        "docket_id": 5068645,
                        "caseName": "Koulkina v. City of New York",
                        "court_id": "nysd",
                        "dateFiled": "2013-03-28",
                    }
                ],
            },
        },
        query="Koulkina",
        search_type="o",
    )

    assert result.http_status == 200  # noqa: PLR2004
    assert result.count == 1
    assert result.next_cursor == "next-page"
    assert result.cache == "miss"
    assert result.records[0].cluster_id == "1969711"
    assert result.records[0].docket_id == "5068645"
    assert result.records[0].case_name == "Koulkina v. City of New York"


def test_recap_search_normalizes_nested_documents() -> None:
    """RECAP results expose validated nested document records."""
    result = normalize_search_payload(
        {
            "status": 200,
            "response": {
                "count": 1,
                "results": [
                    {
                        "docket_id": 42,
                        "caseName": "Doe v. Roe",
                        "recap_documents": [
                            {
                                "id": 88,
                                "entryNumber": 4,
                                "description": "Memorandum Opinion",
                                "entryDateFiled": "2020-06-01",
                                "filepath_local": "recap/example.pdf",
                                "page_count": 24,
                            }
                        ],
                    }
                ],
            },
        },
        query="docket_id:42",
        search_type="r",
    )

    assert result.records[0].docket_id == "42"
    document = result.records[0].recap_documents[0]
    assert document.recap_document_id == "88"
    assert document.entry_number == "4"
    assert document.available is True
    assert document.page_count == 24  # noqa: PLR2004


def test_search_boundary_rejects_known_field_coercion() -> None:
    """Known fields remain strict at the untrusted JSON boundary."""
    with pytest.raises(ValidationError):
        normalize_search_payload(
            {"status": 200, "response": {"count": "1", "results": []}},
            query="Example",
            search_type="o",
        )


def test_search_service_round_trip_preserves_explicit_extra_data() -> None:
    """Service serialization retains explicit response and record extras."""
    original = normalize_search_payload(
        {
            "status": 200,
            "request_id": "request-1",
            "response": {
                "count": 1,
                "search_metadata": {"backend": "solr"},
                "results": [
                    {
                        "id": 1,
                        "caseName": "Example",
                        "score": 9.5,
                    }
                ],
            },
        },
        query="Example",
        search_type="o",
    )

    restored = normalize_search_payload(
        search_result_dict(original),
        query="Example",
        search_type="o",
    )

    assert restored.extra_data == original.extra_data
    assert restored.records[0].extra_data == original.records[0].extra_data


def test_search_failure_is_a_typed_result() -> None:
    """Structured service failures normalize into the same result type."""
    result = normalize_search_payload(
        {"http_status": 503, "detail": "upstream search unavailable"},
        query="Example",
        search_type="o",
    )

    assert result.http_status == 503  # noqa: PLR2004
    assert result.count is None
    assert result.records == ()
    assert result.error_message == "upstream search unavailable"
