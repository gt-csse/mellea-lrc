"""Citation-level aggregation of field assessments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from mellea_lrc.assessment.context import DocumentTextWindow
from mellea_lrc.assessment.fields.case_name import (
    assess_case_name_exact_match,
    assess_case_name_with_mellea,
)
from mellea_lrc.assessment.fields.court import assess_court
from mellea_lrc.jurisdiction_inference.leads import evaluate_reporter_lead, evaluate_court_lead
from mellea_lrc.jurisdiction_inference.types import JurisdictionInference, ReporterLead, CourtLead, ReporterLeadStatus, CourtLeadStatus
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
    reporter: str | None = None,
    session: MelleaSession | None = None,
) -> CitationAssessmentResult:
    """Assess case-name, court, and year fields of one found case citation."""
    reporter_lead = evaluate_reporter_lead(reporter)

    # Court assessment logic: prefer extracted court, fallback to exhaustive reporter inference
    court_lead = evaluate_court_lead(extracted_court)

    # Court assessment logic: prefer extracted court, fallback to exhaustive reporter inference
    court_to_assess = extracted_court
    source: Literal["direct", "reporter_inferred"] = "direct"

    # We do not use translation layer for inference anymore, but reporter_lead could be used in the future
    
    court = assess_court(
        extracted_court=court_to_assess,
        courtlistener_court_id=courtlistener_court_id,
        source=source,
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
        )
    return CitationAssessmentResult(
        case_name=case_name,
        jurisdiction_inference=JurisdictionInference(reporter_lead=evaluate_reporter_lead(reporter) if reporter else ReporterLead(reporter=None, status=ReporterLeadStatus.UNRECOGNIZED, mlz_jurisdictions=()), court_lead=evaluate_court_lead(extracted_court) if extracted_court else CourtLead(extracted_court=None, status=CourtLeadStatus.UNRECOGNIZED, cl_court_taxonomy=None)),
        court=court,
        year=year,
    )
