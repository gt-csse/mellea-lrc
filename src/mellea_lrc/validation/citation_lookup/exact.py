"""Exact CourtListener citation-locator lookup operation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.courtlistener import CourtListenerError
from mellea_lrc.validation.types import (
    ExactLocatorLookupNode,
    LocatorLookupOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.validation.types import CitationValidation

HTTP_OK = 200
HTTP_MULTIPLE_CHOICES = 300
HTTP_NOT_FOUND = 404


def run_exact_locator_lookup(
    validation: CitationValidation,
    *,
    client: CourtListenerServiceClient,
) -> ExactLocatorLookupNode:
    """Return one completed exact-locator validation node."""
    extracted = validation.citation
    citation = extracted.citation
    node_id = f"{extracted.citation_id}:exact_locator_lookup"
    if not isinstance(citation, FullCaseCitation):
        return ExactLocatorLookupNode(
            node_id=node_id,
            status=ValidationNodeStatus.SKIPPED,
            outcome=LocatorLookupOutcome.UNSUPPORTED_CITATION,
            locator=None,
            status_message="Skipped exact locator lookup because this is not a full case citation.",
            outcome_message="Citation is not a full case citation with a reporter locator.",
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
            status_message="Skipped exact locator lookup because the locator is incomplete.",
            outcome_message="Citation is missing volume, reporter, or page needed for locator lookup.",
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
            status_message="Exact locator lookup failed while calling CourtListener.",
            outcome_message="CourtListener locator lookup could not be completed.",
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
            status_message="Exact locator lookup completed.",
            outcome_message="CourtListener returned one citation record for the locator.",
        )
    if lookup.status == HTTP_NOT_FOUND and not lookup.records:
        return ExactLocatorLookupNode(
            node_id=node_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=LocatorLookupOutcome.NOT_FOUND,
            locator=locator,
            status_message="Exact locator lookup completed.",
            outcome_message="CourtListener returned no citation records for the locator.",
        )
    if lookup.status == HTTP_MULTIPLE_CHOICES and len(lookup.records) > 1:
        return ExactLocatorLookupNode(
            node_id=node_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=LocatorLookupOutcome.AMBIGUOUS,
            locator=locator,
            candidate_count=len(lookup.records),
            status_message="Exact locator lookup completed.",
            outcome_message=(
                f"CourtListener returned {len(lookup.records)} citation records for the locator."
            ),
        )
    msg = f"Unexpected CourtListener lookup response: status={lookup.status}, records={len(lookup.records)}"
    raise AssertionError(msg)
