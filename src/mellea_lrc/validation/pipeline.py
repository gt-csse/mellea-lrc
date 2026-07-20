"""Exact-locator validation pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.courtlistener import CourtListenerClient, CourtListenerError
from mellea_lrc.validation.model import (
    CitationValidation,
    ExactLocatorLookupNode,
    LocatorLookupOutcome,
    ValidationDocument,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument

HTTP_OK = 200
HTTP_NOT_FOUND = 404


def initialize_validation(document: ExtractedDocument) -> ValidationDocument:
    """Create one empty validation progression per extracted citation."""
    return ValidationDocument(
        source=document,
        citations=tuple(CitationValidation(citation=item) for item in document.citations),
    )


def validate_exact_locators(
    document: ExtractedDocument,
    *,
    client: CourtListenerServiceClient | None = None,
) -> ValidationDocument:
    """Run only the exact locator lookup and retain only its found branch."""
    service = client if client is not None else CourtListenerClient()
    initialized = initialize_validation(document)
    citations = tuple(item.append(_lookup_locator(item.citation, service)) for item in initialized.citations)
    return ValidationDocument(source=document, citations=citations)


def _lookup_locator(
    extracted: ExtractedCitation,
    client: CourtListenerServiceClient,
) -> ExactLocatorLookupNode:
    citation = extracted.citation
    node_id = f"{extracted.citation_id}:exact_locator_lookup"
    if not isinstance(citation, FullCaseCitation):
        return ExactLocatorLookupNode(
            node_id=node_id,
            status=ValidationNodeStatus.SKIPPED,
            outcome=LocatorLookupOutcome.UNSUPPORTED_CITATION,
            locator=None,
        )

    volume = citation.volume
    reporter = citation.reporter
    page = citation.page
    if not volume or not reporter or not page:
        return ExactLocatorLookupNode(
            node_id=node_id,
            status=ValidationNodeStatus.SKIPPED,
            outcome=LocatorLookupOutcome.INCOMPLETE_LOCATOR,
            locator=None,
        )

    locator = f"{volume} {reporter} {page}"
    try:
        lookup = client.lookup_citation(volume, reporter, page)
    except CourtListenerError as exc:
        return ExactLocatorLookupNode(
            node_id=node_id,
            status=ValidationNodeStatus.FAILED,
            outcome=LocatorLookupOutcome.FAILED,
            locator=locator,
            error=str(exc),
        )

    if lookup.status == HTTP_OK and len(lookup.records) == 1:
        return ExactLocatorLookupNode(
            node_id=node_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=LocatorLookupOutcome.FOUND,
            locator=locator,
            record=lookup.records[0],
            candidate_count=1,
        )
    if lookup.status == HTTP_NOT_FOUND or not lookup.records:
        outcome = LocatorLookupOutcome.NOT_FOUND
    else:
        outcome = LocatorLookupOutcome.AMBIGUOUS
    return ExactLocatorLookupNode(
        node_id=node_id,
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=outcome,
        locator=locator,
        candidate_count=len(lookup.records),
    )
