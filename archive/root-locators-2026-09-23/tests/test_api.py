"""Tests for the outer compositional API."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from mellea_lrc.api import (
    Document,
    find_docket_locators,
    find_full_reporter_locators,
    form_roots,
    grow_leaves,
    grow_roots,
    lookup_full_reporter_locators_exact,
    resolve_colocations,
    resolve_full_reporter_locator_ambiguities,
    validate_roots_identity,
    validate_unique_full_reporter_locator_identities,
)
from mellea_lrc.courtlistener import CourtListenerCitationLookup, CourtListenerSearchResult
from mellea_lrc.extraction.eyecite_extractor import extract_from_plain_text
from mellea_lrc.govinfo import GovInfoSearchResult
from mellea_lrc.model.citations import DocketCitation, FullCaseCitation, placed
from mellea_lrc.model.record import CitationRecord
from mellea_lrc.model.spans import Span
from tests.record_fixtures import read_citation


def test_document_owns_its_serialization_before_and_after_identity() -> None:
    document = extract_from_plain_text("A filing without citations.")

    recovered_document = Document.model_validate(document.model_dump(mode="json"))
    lookup = asyncio.run(lookup_full_reporter_locators_exact(form_roots(recovered_document), client=object()))
    unique = asyncio.run(validate_unique_full_reporter_locator_identities(lookup, client=object()))
    completed = asyncio.run(resolve_full_reporter_locator_ambiguities(unique, client=object()))
    restored_completed = Document.model_validate(completed.model_dump(mode="json"))

    assert recovered_document == document
    assert restored_completed == completed


def test_outer_api_composes_locator_stages_from_a_document_owned_constructor() -> None:
    document = Document.from_source(
        "Smith v. Jones, No. 1:24-cv-00123, 2024 WL 1234567 (D. Ariz. Jan. 1, 2024)."
    )

    document = find_full_reporter_locators(document)
    document = find_docket_locators(document)
    document = resolve_colocations(document)

    assert len(document.citations) == 2
    assert all(citation.colocation_id is not None for citation in document.citations)


def test_document_from_source_preserves_a_path_as_source_provenance(tmp_path: Path) -> None:
    path = tmp_path / "filing.txt"
    path.write_text("A filing without citations.", encoding="utf-8")

    document = Document.from_source(path)

    assert document.source_path == str(path)


class _OrderedNoResultClient:
    """Records the two independent root retrieval routes without model work."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str, search_type: str, cursor: str | None = None, *, semantic: bool = False):
        del cursor
        self.calls.append(f"docket:{query}:{search_type}")
        return CourtListenerSearchResult.from_payload(
            query=query,
            search_type=search_type,
            semantic=semantic,
            count=0,
            results=[],
            next_cursor=None,
            previous_cursor=None,
        )

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        self.calls.append(f"reporter:{volume} {reporter} {page}")
        return CourtListenerCitationLookup(citation=f"{volume} {reporter} {page}", status=404, clusters=())


class _NoResultGovInfoClient:
    """Avoid live GovInfo traffic while proving the composition boundary."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search_uscourts(self, query: str, *, page_size: int):
        self.calls.append((query, page_size))
        return GovInfoSearchResult(
            query=query,
            count=0,
            results=(),
            next_offset_mark=None,
        )


def test_root_identity_composition_preserves_docket_query_plan_and_reporter_checkpoints() -> None:
    text = "x" * 200
    document = Document.from_plain_text(text)
    docket = read_citation(
        citation_id="docket",
        fields=placed(
            DocketCitation(docket_number="1:24-cv-00123"),
            span=Span(10, 26),
            locator_span=Span(10, 26),
            matched_text="1:24-cv-00123",
        ),
    )
    reporter = read_citation(
        citation_id="reporter",
        fields=placed(
            FullCaseCitation(volume="347", reporter="U.S.", page="483"),
            span=Span(50, 62),
            locator_span=Span(50, 62),
            matched_text="347 U.S. 483",
        ),
    )
    client = _OrderedNoResultClient()
    govinfo_client = _NoResultGovInfoClient()

    completed = asyncio.run(
        validate_roots_identity(
            form_roots(document.evolve(citations=(docket, reporter))),
            client=client,
            govinfo_client=govinfo_client,
        )
    )

    assert client.calls == ["docket:1:24-cv-00123:d", "reporter:347 U.S. 483"]
    assert govinfo_client.calls == [('collection:uscourts casenumber:("1:24-cv-00123")', 100)]
    assert completed.passes[-5:] == (
        "docket_root_identity",
        "full_reporter_locator_exact_lookup",
        "full_reporter_locator_unique_identity",
        "full_reporter_exact_date_review",
        "full_reporter_locator_ambiguity_resolution",
    )


def test_root_and_leaf_composition_can_run_without_identity_validation() -> None:
    document = Document.from_source("Bell Atlantic Corp. v. Twombly, 550 U.S. 544 (2007). Id. at 570.")

    roots = asyncio.run(grow_roots(document))
    grown = asyncio.run(grow_leaves(roots))

    assert "root_formation" in roots.passes
    assert grown.passes[-1] == "leaf_growth"
    assert any(
        citation.root_id is not None and citation.citation_id != citation.root_id
        for citation in grown.citations
    )


def test_compositional_leaf_growth_requires_explicit_root_formation() -> None:
    with pytest.raises(ValueError, match="Leaf growth requires root_formation"):
        asyncio.run(grow_leaves(Document.from_source("Id. at 570.")))
