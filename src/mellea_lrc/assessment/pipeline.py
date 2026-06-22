"""Document-level assessment pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.assessment.case_name import assess_case_name_exact_match, build_extracted_case_name
from mellea_lrc.assessment.citation import assess_year_exact_match
from mellea_lrc.assessment.context import find_text_span_near_full_span, get_extended_span_text
from mellea_lrc.assessment.types import (
    CaseNameAssessmentStatus,
    CitationAssessment,
    DocumentAssessment,
    ModifiedExtractedCitation,
)
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.llm import start_mellea_session_from_env
from mellea_lrc.validation.types import ValidationStatus

if TYPE_CHECKING:
    from collections.abc import Callable

    from mellea_lrc.validation.types import CitationValidation, DocumentValidation


@dataclass(frozen=True, slots=True)
class MelleaCallContext:
    """Context emitted immediately before one semantic assessment call."""

    mellea_call: int
    citation_id: str
    matched_text: str
    extracted_case_name: str | None
    courtlistener_case_name: str | None
    context: str


def run_assessment(
    validation: DocumentValidation,
    *,
    max_mellea: int | None = None,
    on_mellea_call: Callable[[MelleaCallContext], None] | None = None,
) -> DocumentAssessment:
    """Assess a validated document.

    The interface boundary is intentionally simple:
    :class:`DocumentValidation` in, :class:`DocumentAssessment` out.
    Deterministic exact-match and year checks always run. Mellea-backed semantic
    assessment is called only for case names that need it.
    """
    validations_by_id = {item.citation_id: item for item in validation.validations}
    assessments: list[CitationAssessment] = []
    modified_citations: list[ModifiedExtractedCitation] = []
    reassessments: list[CitationAssessment] = []
    session = None
    mellea_calls = 0

    for citation in validation.citations:
        citation_validation = validations_by_id.get(citation.citation_id)
        if citation_validation is None:
            continue
        if not isinstance(citation.citation, FullCaseCitation):
            continue
        if citation_validation.status != ValidationStatus.FOUND:
            continue

        extracted_case_name = build_extracted_case_name(citation.citation)
        courtlistener_case_name = _first_cluster_case_name(citation_validation)
        year_assess = assess_year_exact_match(
            citation_id=citation.citation_id,
            extracted_year=citation.citation.year,
            courtlistener_year=_first_cluster_year(citation_validation),
        )
        exact = assess_case_name_exact_match(
            citation_id=citation.citation_id,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
        )
        if exact.status != CaseNameAssessmentStatus.NEEDS_ASSESSMENT:
            assessments.append(
                CitationAssessment(
                    citation_id=citation.citation_id,
                    case_assess=exact,
                    year_assess=year_assess,
                )
            )
            continue
        if max_mellea is not None and mellea_calls >= max_mellea:
            continue

        session = session or start_mellea_session_from_env()
        from mellea_lrc.assessment.mellea import assess_case_name_with_mellea  # noqa: PLC0415

        context = get_extended_span_text(validation.text, citation.span)
        if on_mellea_call is not None:
            on_mellea_call(
                MelleaCallContext(
                    mellea_call=mellea_calls + 1,
                    citation_id=citation.citation_id,
                    matched_text=citation.matched_text,
                    extracted_case_name=extracted_case_name,
                    courtlistener_case_name=courtlistener_case_name,
                    context=context,
                )
            )
        run = assess_case_name_with_mellea(
            session,
            citation_id=citation.citation_id,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            document_context=context,
        )
        mellea_calls += 1
        assessments.append(
            CitationAssessment(
                citation_id=citation.citation_id,
                case_assess=run.assessment,
                year_assess=year_assess,
            )
        )
        if run.modified_citation is not None:
            modified_citations.append(
                ModifiedExtractedCitation.from_proposal(
                    run.modified_citation,
                    citation_id=citation.citation_id,
                    span=find_text_span_near_full_span(
                        validation.text,
                        run.modified_citation.extracted_case_name or "",
                        citation.span,
                    ),
                )
            )
        if run.reassessment is not None:
            reassessments.append(
                CitationAssessment(
                    citation_id=citation.citation_id,
                    case_assess=run.reassessment,
                    year_assess=year_assess,
                )
            )

    return DocumentAssessment(
        preprocessed=validation.preprocessed,
        citations=validation.citations,
        validations=validation.validations,
        assessments=tuple(assessments),
        modified_citations=tuple(modified_citations),
        reassessments=tuple(reassessments),
    )


def _first_cluster_case_name(validation: CitationValidation) -> str | None:
    if not validation.clusters:
        return None
    case_name = validation.clusters[0].get("case_name") or validation.clusters[0].get("caseName")
    return str(case_name) if isinstance(case_name, str) and case_name else None


def _first_cluster_year(validation: CitationValidation) -> str | None:
    if not validation.clusters:
        return None
    date_filed = validation.clusters[0].get("date_filed") or validation.clusters[0].get("dateFiled")
    return str(date_filed)[:4] if isinstance(date_filed, str) and date_filed else None
