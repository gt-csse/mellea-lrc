"""Citation-level aggregation of field assessments."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.assessment.context import DocumentTextWindow
from mellea_lrc.assessment.fields.case_name import (
    assess_case_name_exact_match,
    assess_case_name_with_mellea,
)
from mellea_lrc.assessment.fields.court import assess_court
from mellea_lrc.assessment.fields.year import assess_year_exact_match
from mellea_lrc.assessment.types import (
    CaseNameAssessmentRun,
    CaseNameReassessmentNotRequired,
    CitationAssessmentResult,
)

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.core.spans import Span


async def assess_found_citation(
    *,
    document_text: str,
    span: Span,
    extracted_case_name: str | None,
    courtlistener_case_name: str | None,
    extracted_year: str | None,
    courtlistener_year: str | None,
    extracted_court: str | None = None,
    courtlistener_court_id: str | None = None,
    citation_locator: str | None = None,
    session: MelleaSession | None = None,
) -> CitationAssessmentResult:
    """Assess case-name, court, and year fields of one found case citation."""
    court = assess_court(
        extracted_court=extracted_court,
        courtlistener_court_id=courtlistener_court_id,
        source="direct",
    )

    year = assess_year_exact_match(
        extracted_year=extracted_year,
        courtlistener_year=courtlistener_year,
    )
    exact = assess_case_name_exact_match(
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
    )
    if exact is not None:
        case_name = CaseNameAssessmentRun(
            initial=exact,
            followup=CaseNameReassessmentNotRequired(),
        )
    else:
        if session is None:
            msg = "A Mellea session is required for a non-exact case-name assessment"
            raise RuntimeError(msg)
        case_name = await assess_case_name_with_mellea(
            session,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            document_context=DocumentTextWindow.around(document_text, span),
            citation_locator=citation_locator,
        )
    return CitationAssessmentResult(
        case_name=case_name,
        court=court,
        year=year,
    )
