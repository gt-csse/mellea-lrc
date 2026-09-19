"""Contract tests for the independent complete-locator root graph stage."""

from __future__ import annotations

import asyncio

from mellea_lrc.api import (
    form_roots,
    lookup_full_reporter_locators_exact,
    resolve_full_reporter_locator_ambiguities,
    validate_unique_full_reporter_locator_identities,
)
from mellea_lrc.core.citations import DocketCitation, FullCaseCitation, placed
from mellea_lrc.core.record import CitationRecord, Question
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener import CourtListenerCitationLookup
from mellea_lrc.extraction import Document, ExtractionMetadata
from mellea_lrc.preprocessing import preprocess


def _document(*records: CitationRecord) -> Document:
    text = "x" * 200
    preprocessed = preprocess(text)
    return Document(
        source_metadata=preprocessed.source_metadata,
        text=text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=records,
        extraction_metadata=ExtractionMetadata(),
    )


def _reporter_record(citation_id: str, start: int, *, volume: str = "347") -> CitationRecord:
    locator = f"{volume} U.S. 483"
    return CitationRecord(
        citation_id=citation_id,
        source=placed(
            FullCaseCitation(volume=volume, reporter="U.S.", page="483"),
            span=Span(start, start + len(locator)),
            locator_span=Span(start, start + len(locator)),
            matched_text=locator,
        ),
    )


def _docket_record(citation_id: str, start: int, *, court: str | None) -> CitationRecord:
    locator = "No. 1:24-cv-00123"
    return CitationRecord(
        citation_id=citation_id,
        source=placed(
            DocketCitation(docket_number="1:24-cv-00123", court=court),
            span=Span(start, start + len(locator)),
            locator_span=Span(start, start + len(locator)),
            matched_text=locator,
        ),
    )


def test_form_roots_attaches_only_exact_repeated_complete_identifiers() -> None:
    first = _reporter_record("reporter-1", 10)
    repeated = _reporter_record("reporter-2", 40)
    parallel = _reporter_record("reporter-3", 70, volume="348")
    parallel.colocation_id = "colocation-1"
    first.colocation_id = "colocation-1"

    formed = form_roots(_document(first, repeated, parallel))
    first_result, repeated_result, parallel_result = formed.citations

    assert first_result.root_id == "reporter-1"
    assert repeated_result.root_id == "reporter-1"
    assert repeated_result.resolves_to == "reporter-1"
    assert parallel_result.root_id == "reporter-3"
    assert first_result.trace[-1].outcome == "root_created"
    assert repeated_result.trace[-1].outcome == "root_attached"
    assert formed.passes[-1] == "root_formation"
    assert Document.from_serialized(formed.serialize()) == formed
    assert form_roots(formed) is formed


def test_form_roots_only_deduplicates_dockets_with_a_stated_court() -> None:
    court_first = _docket_record("docket-1", 10, court="nyed")
    court_repeat = _docket_record("docket-2", 40, court="nyed")
    courtless_first = _docket_record("docket-3", 70, court=None)
    courtless_repeat = _docket_record("docket-4", 100, court=None)

    formed = form_roots(_document(court_first, court_repeat, courtless_first, courtless_repeat))

    assert [record.root_id for record in formed.citations] == [
        "docket-1",
        "docket-1",
        "docket-3",
        "docket-4",
    ]


class _NoResultClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        self.calls.append((volume, reporter, page))
        return CourtListenerCitationLookup(citation=f"{volume} {reporter} {page}", status=404, clusters=())


def test_identity_reads_each_formed_reporter_root_once() -> None:
    client = _NoResultClient()
    document = form_roots(_document(_reporter_record("first", 10), _reporter_record("repeat", 40)))

    lookup = asyncio.run(lookup_full_reporter_locators_exact(document, client=client))
    unique = asyncio.run(validate_unique_full_reporter_locator_identities(lookup, client=client))
    completed = asyncio.run(resolve_full_reporter_locator_ambiguities(unique, client=client))

    assert client.calls == [("347", "U.S.", "483")]
    assert completed.citations[0].judgement(Question.LOCATOR_LOOKUP).outcome == "deferred_to_search"
    assert completed.citations[0].judgement(Question.IDENTITY).outcome == "unjudged"
    assert completed.citations[1].judgement(Question.IDENTITY).outcome == "unjudged"
