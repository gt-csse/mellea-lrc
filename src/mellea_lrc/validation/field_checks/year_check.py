"""Decision-year field validation."""

from __future__ import annotations

from datetime import date as CalendarDate
from typing import TYPE_CHECKING

from mellea_lrc.core.citations import CitationDate, DocketCitation, FullCaseCitation
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
    citation = validation.citation.stated
    date = citation.date if isinstance(citation, (FullCaseCitation, DocketCitation)) else None
    # The check compares years; a citation stating a full date states its year too.
    extracted = date.year if date else None
    retrieved = candidate.year
    if isinstance(citation, DocketCitation) and candidate.source is CandidateEvaluationSource.DOCKET_SEARCH:
        outcome = _docket_filing_date_outcome(date, candidate.date_filed)
        status = (
            ValidationNodeStatus.SKIPPED
            if outcome is FieldCheckOutcome.UNAVAILABLE
            else ValidationNodeStatus.SUCCEEDED
        )
        status_message = (
            "Skipped docket filing-date chronology comparison because required evidence is missing."
            if outcome is FieldCheckOutcome.UNAVAILABLE
            else "Docket filing-date chronology comparison completed."
        )
        outcome_message = {
            FieldCheckOutcome.MATCH: (
                "The CourtListener case filing date is on or before the citation's stated decision date."
            ),
            FieldCheckOutcome.MISMATCH: (
                "The CourtListener case filing date is after the citation's stated decision date."
            ),
            FieldCheckOutcome.UNAVAILABLE: (
                "Docket filing-date chronology is unavailable because one date is missing or unreadable."
            ),
        }[outcome]
        # TODO: Retrieve a dated order or opinion from the resolved docket and
        # compare that decision's date directly. Filing-date chronology is only
        # a compatibility check; it does not prove that a particular order exists.
    elif extracted is None or retrieved is None:
        status = ValidationNodeStatus.SKIPPED
        status_message = "Skipped year comparison because required evidence is missing."
        outcome = FieldCheckOutcome.UNAVAILABLE
        outcome_message = "Year comparison is unavailable because one year is missing."
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
        depends_on=(candidate.node_id,),
        status_message=status_message,
        outcome_message=outcome_message,
    )


def _docket_filing_date_outcome(
    citation_date: CitationDate | None,
    filing_date: str | None,
) -> FieldCheckOutcome:
    """Check only whether a case began no later than its cited decision.

    A docket's ``dateFiled`` and an opinion date name different events. They
    are nevertheless chronologically compatible when the case filing date is
    not later than the decision date. A year-only citation supplies only a
    year boundary, so equality is not treated as contradictory.
    """
    if citation_date is None or filing_date is None:
        return FieldCheckOutcome.UNAVAILABLE
    filed = _parse_filing_date(filing_date)
    if filed is None:
        return FieldCheckOutcome.UNAVAILABLE
    decision = _parse_citation_date(citation_date)
    if decision is not None:
        return FieldCheckOutcome.MATCH if filed <= decision else FieldCheckOutcome.MISMATCH
    try:
        decision_year = int(citation_date.year)
    except ValueError:
        return FieldCheckOutcome.UNAVAILABLE
    return FieldCheckOutcome.MATCH if filed.year <= decision_year else FieldCheckOutcome.MISMATCH


def _parse_filing_date(value: str) -> CalendarDate | None:
    """Read CourtListener's ISO ``dateFiled`` form without coercing variants."""
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
