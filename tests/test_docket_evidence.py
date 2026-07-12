"""Tests for bounded document-level evidence expansion from RECAP dockets."""

# ruff: noqa: INP001

from dataclasses import dataclass, field

from mellea_lrc.retrieval.docket_evidence import expand_docket_evidence
from mellea_lrc.retrieval.types import DocketEvidenceStatus


@dataclass
class _EvidenceClient:
    queries: list[str] = field(default_factory=list)

    def get_docket(self, docket_id: str) -> dict[str, object]:
        assert docket_id == "42"
        return {
            "http_status": 200,
            "cache": "hit",
            "key": "docket-42",
            "case_name": "Doe v. Roe",
            "court_id": "nmd",
            "docket_number": "1:19-cv-00042",
            "date_filed": "2019-02-01",
            "date_terminated": "2021-04-02",
            "assigned_to_str": "Judge Example",
            "nature_of_suit": "Civil Rights",
            "cause": "42:1983",
            "jurisdiction_type": "Federal Question",
        }

    def search_recap(self, query: str) -> dict[str, object]:
        self.queries.append(query)
        return {
            "http_status": 200,
            "cache": "miss",
            "key": "documents-42-2020",
            "results": [
                {
                    "recap_documents": [
                        {
                            "id": 11,
                            "entry_number": "50",
                            "description": "Minute order",
                            "entry_date_filed": "2020-06-02",
                            "available": True,
                        },
                        {
                            "id": 12,
                            "entry_number": "49",
                            "description": "Memorandum Opinion and Order",
                            "entry_date_filed": "2020-06-01",
                            "page_count": 24,
                            "pacer_doc_id": "9001",
                            "available": False,
                        },
                        {
                            "id": 13,
                            "entry_number": "51",
                            "description": "Exhibit",
                            "entry_date_filed": "2020-06-03",
                            "available": True,
                        },
                    ]
                }
            ],
        }


def test_docket_evidence_uses_cited_year_and_ranks_specific_decisional_cues() -> None:
    """A year-scoped memorandum opinion outranks a generic available order."""
    client = _EvidenceClient()

    evidence = expand_docket_evidence(client=client, docket_id="42", cited_year="2020")

    assert client.queries == [
        "docket_id:42 AND entry_date_filed:[2020-01-01 TO 2020-12-31] AND "
        '("memorandum opinion" OR opinion OR decision OR judgment OR order)'
    ]
    assert evidence.status is DocketEvidenceStatus.ENRICHED
    assert evidence.case_name == "Doe v. Roe"
    assert evidence.assigned_to == "Judge Example"
    assert evidence.documents[0].recap_document_id == "12"
    assert evidence.documents[0].decisional_cues == ("memorandum_opinion", "opinion", "order")
    assert evidence.documents[0].year_distance == 0
    assert evidence.documents[1].recap_document_id == "11"
    assert all(document.recap_document_id != "13" for document in evidence.documents)


def test_docket_evidence_uses_full_date_as_a_soft_ranking_cue() -> None:
    """The asserted date wins when present but does not exclude nearby documents."""
    client = _EvidenceClient()

    evidence = expand_docket_evidence(
        client=client,
        docket_id="42",
        cited_year="2020",
        cited_date="2020-06-02",
    )

    assert "entry_date_filed:[2020-06-02 TO 2020-06-02]" in client.queries[0]
    assert evidence.documents[0].recap_document_id == "11"


def test_docket_evidence_retries_with_flexible_year_only_after_empty_exact_search() -> None:
    """Nearby years are queried only when the precise year has no decision."""
    client = _EvidenceClient()
    responses = iter(
        (
            {"http_status": 200, "results": []},
            {
                "http_status": 200,
                "results": [
                    {
                        "recap_documents": [
                            {
                                "id": 14,
                                "description": "Opinion",
                                "entry_date_filed": "2021-01-02",
                            }
                        ]
                    }
                ],
            },
        )
    )

    def search(query: str) -> dict[str, object]:
        client.queries.append(query)
        return next(responses)

    client.search_recap = search  # type: ignore[method-assign]
    evidence = expand_docket_evidence(client=client, docket_id="42", cited_year="2020")

    assert len(client.queries) == 2  # noqa: PLR2004 - exact request plus one fallback
    assert "[2020-01-01 TO 2020-12-31]" in client.queries[0]
    assert "[2019-01-01 TO 2021-12-31]" in client.queries[1]
    assert evidence.documents[0].year_distance == 1


def test_docket_evidence_reports_no_decisional_documents_without_guessing() -> None:
    """A successful bounded search with no cues remains evidence, not a match."""
    client = _EvidenceClient()
    client.search_recap = lambda _query: {  # type: ignore[method-assign]
        "http_status": 200,
        "results": [{"recap_documents": [{"description": "Exhibit", "available": True}]}],
    }

    evidence = expand_docket_evidence(client=client, docket_id="42", cited_year="2020")

    assert evidence.status is DocketEvidenceStatus.NO_DECISIONAL_DOCUMENTS
    assert evidence.documents == ()


def test_docket_evidence_is_unavailable_without_expansion_methods() -> None:
    """Clients lacking docket expansion degrade explicitly and safely."""
    evidence = expand_docket_evidence(client=object(), docket_id="42", cited_year="2020")  # type: ignore[arg-type]

    assert evidence.status is DocketEvidenceStatus.UNAVAILABLE

