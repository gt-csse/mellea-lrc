"""Decision-year field validation."""

from __future__ import annotations

from datetime import date as CalendarDate
from typing import TYPE_CHECKING

from mellea_lrc.model.citations import CitationDate, DocketCitation, FullCaseCitation
from mellea_lrc.validation.types import (
    CandidateEvaluationSource,
    FieldCheckOutcome,
    ValidationNodeStatus,
    YearCheckNode,
)

if TYPE_CHECKING:
    from mellea_lrc.validation.types import CandidateEvaluationNode, CitationValidation


def run_year_check(
    validation: CitationValidation,
    *,
    candidate: CandidateEvaluationNode,
) -> YearCheckNode:
    """Compare a citation date with the date semantics of one retrieval route."""
    citation = validation.citation.fields
    date = citation.date if isinstance(citation, (FullCaseCitation, DocketCitation)) else None
    extracted = date.year if date else None
    retrieved = candidate.year
    parsed_citation_date = _parse_citation_date(date) if date else None
    extracted_date = parsed_citation_date.isoformat() if parsed_citation_date else None
    docket_sources = {
        CandidateEvaluationSource.DOCKET_SEARCH,
        CandidateEvaluationSource.COURTLISTENER_DOCKET_BODY_SEARCH,
        CandidateEvaluationSource.FULL_REPORTER_METADATA_SEARCH,
        CandidateEvaluationSource.RECAP_SEARCH,
    }
    govinfo_sources = {
        CandidateEvaluationSource.GOVINFO_DOCKET_SEARCH,
        CandidateEvaluationSource.GOVINFO_BODY_SEARCH,
        CandidateEvaluationSource.GOVINFO_FULL_REPORTER_METADATA_SEARCH,
    }
    opinion_sources = {
        CandidateEvaluationSource.LOCATOR_LOOKUP,
        CandidateEvaluationSource.OPINION_SEARCH,
        CandidateEvaluationSource.COURTLISTENER_CLUSTER_BODY_SEARCH,
    }
    if candidate.source in docket_sources:
        retrieved_date = candidate.date_filed
    elif candidate.source in govinfo_sources:
        retrieved_date = None
    else:
        retrieved_date = candidate.decision_date or candidate.date_filed
    if candidate.source in docket_sources:
        outcome = _docket_filing_date_outcome(date, candidate.date_filed)
        status = ValidationNodeStatus.SKIPPED if outcome is None else ValidationNodeStatus.SUCCEEDED
        status_message = (
            "No decision-date judgment follows from a compatible or missing case filing date."
            if outcome is None
            else "Docket filing-date chronology comparison completed."
        )
        outcome_message = (
            "The CourtListener case filing date is after the citation's stated decision date."
            if outcome is FieldCheckOutcome.MISMATCH
            else "The case filing date cannot establish the cited decision date; no date judgment was made."
        )
        # TODO: Retrieve a dated order or opinion from the resolved docket and
        # compare that decision's date directly. Filing-date chronology can
        # rule out a candidate, but it never proves the cited order exists on
        # that date.
    elif candidate.source in govinfo_sources:
        # A GovInfo USCOURTS package represents the case rather than the one
        # order cited in the filing. ``packageDateIssued`` is useful provenance
        # but cannot confirm or contradict the citation date.
        status = ValidationNodeStatus.SKIPPED
        outcome = None
        status_message = "Skipped GovInfo package-date comparison."
        outcome_message = (
            "The GovInfo package date identifies a deposited record, not necessarily the cited order."
        )
    elif extracted is None or retrieved is None:
        status = ValidationNodeStatus.SKIPPED
        status_message = "Skipped year comparison because required evidence is missing."
        outcome = None
        outcome_message = "No year judgment was made because one side has no date."
    elif (
        isinstance(citation, FullCaseCitation)
        and candidate.source in opinion_sources
        and extracted_date is not None
    ):
        # These routes date the opinion itself. Docket ``dateFiled`` instead
        # dates the case, so it is handled only by the chronology branch above.
        opinion_date = _parse_filing_date(retrieved_date) if retrieved_date else None
        if opinion_date is not None:
            status = ValidationNodeStatus.SUCCEEDED
            outcome = (
                FieldCheckOutcome.MATCH
                if extracted_date == opinion_date.isoformat()
                else FieldCheckOutcome.MISMATCH
            )
            status_message = "Full decision-date comparison completed."
            outcome_message = (
                "Cited and retrieved decision dates match."
                if outcome is FieldCheckOutcome.MATCH
                else "Cited and retrieved decision dates differ."
            )
        else:
            status = ValidationNodeStatus.SUCCEEDED
            outcome = FieldCheckOutcome.MATCH if extracted == retrieved else FieldCheckOutcome.MISMATCH
            status_message = "Year comparison completed because a complete opinion date is unavailable."
            outcome_message = (
                "Cited and retrieved years match."
                if outcome is FieldCheckOutcome.MATCH
                else "Cited and retrieved years differ."
            )
    else:
        status = ValidationNodeStatus.SUCCEEDED
        status_message = "Year comparison completed."
        outcome = FieldCheckOutcome.MATCH if extracted == retrieved else FieldCheckOutcome.MISMATCH
        outcome_message = (
            "Extracted and retrieved years match."
            if outcome is FieldCheckOutcome.MATCH
            else "Extracted and retrieved years differ."
        )
    return YearCheckNode(
        node_id=f"{candidate.node_id}:year_check",
        status=status,
        outcome=outcome,
        extracted_year=extracted,
        retrieved_year=retrieved,
        extracted_date=extracted_date,
        retrieved_date=retrieved_date,
        depends_on=(candidate.node_id,),
        status_message=status_message,
        outcome_message=outcome_message,
    )


def _docket_filing_date_outcome(
    citation_date: CitationDate | None,
    filing_date: str | None,
) -> FieldCheckOutcome | None:
    """Check whether a filing date contradicts a cited decision date.

    A docket's ``dateFiled`` and an opinion date name different events. They
    are nevertheless chronologically compatible when the case filing date is
    not later than the decision date. Compatibility produces no date judgment:
    it cannot prove the particular cited order or opinion.
    A year-only citation supplies only a year boundary, so equality is not
    treated as contradictory.
    """
    if citation_date is None or filing_date is None:
        return None
    filed = _parse_filing_date(filing_date)
    if filed is None:
        return None
    decision = _parse_citation_date(citation_date)
    if decision is not None:
        return None if filed <= decision else FieldCheckOutcome.MISMATCH
    try:
        decision_year = int(citation_date.year)
    except ValueError:
        return None
    return None if filed.year <= decision_year else FieldCheckOutcome.MISMATCH


def _parse_filing_date(value: str) -> CalendarDate | None:
    """Read a provider's complete ISO date without coercing partial dates."""
    try:
        return CalendarDate.fromisoformat(value)
    except ValueError:
        return None


def _parse_citation_date(value: CitationDate) -> CalendarDate | None:
    """Read the full day a citation actually states, if it states one."""
    if value.month is None or value.day is None:
        return None
    month = _MONTH_NUMBERS.get(value.month.casefold().rstrip(".")[:3])
    if month is None:
        return None
    try:
        return CalendarDate(int(value.year), month, int(value.day))
    except ValueError:
        return None


_MONTH_NUMBERS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
