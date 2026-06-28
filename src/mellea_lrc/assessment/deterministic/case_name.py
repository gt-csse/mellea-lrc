"""Deterministic case-name comparison."""

import unicodedata
from collections.abc import Iterable

from mellea_lrc.assessment.types import ChatTurn, CaseNameAssessment, CaseNameAssessmentStatus
from mellea_lrc.core.citations import FullCaseCitation

_TYPOGRAPHIC_TRANSLATION = str.maketrans(
    {
        "\N{LEFT SINGLE QUOTATION MARK}": "'",
        "\N{RIGHT SINGLE QUOTATION MARK}": "'",
        "\N{SINGLE HIGH-REVERSED-9 QUOTATION MARK}": "'",
        "\N{PRIME}": "'",
        "\N{LEFT DOUBLE QUOTATION MARK}": '"',
        "\N{RIGHT DOUBLE QUOTATION MARK}": '"',
        "\N{EN DASH}": "-",
        "\N{EM DASH}": "-",
        "\N{MINUS SIGN}": "-",
    }
)

_STATUS_MESSAGES = {
    CaseNameAssessmentStatus.EXACT_MATCH: "Extracted case name exactly matches CourtListener.",
    CaseNameAssessmentStatus.SEMANTIC_MATCH: "Extracted case name matches the retrieved case.",
    CaseNameAssessmentStatus.DIFFERENT_CASE: (
        "Extracted case name refers to a different case than the retrieved record."
    ),
    CaseNameAssessmentStatus.IRREGULAR_FORM: (
        "Extracted case name uses an unusual or incomplete form for this case."
    ),
    CaseNameAssessmentStatus.REEXTRACTION_FAIL: "Case-name re-extraction failed.",
    CaseNameAssessmentStatus.NEEDS_ASSESSMENT: "Case name has not been assessed.",
}


def normalize_case_name(value: str) -> str:
    """Normalize a case name for equality checks (quotes, dashes, whitespace)."""
    normalized = unicodedata.normalize("NFKC", value).translate(_TYPOGRAPHIC_TRANSLATION)
    return " ".join(normalized.split())


def case_names_equivalent(left: str | None, right: str | None) -> bool:
    """Return whether two case names match once typographic noise is removed."""
    if not left or not right:
        return False
    return normalize_case_name(left) == normalize_case_name(right)


def build_extracted_case_name(citation: FullCaseCitation) -> str | None:
    """Build a display case name from extracted party fields."""
    if citation.plaintiff and citation.defendant:
        return f"{citation.plaintiff} v. {citation.defendant}"
    return citation.plaintiff or citation.defendant


def build_case_name_assessment(
    citation_id: str,
    status: CaseNameAssessmentStatus,
    extracted_case_name: str | None,
    courtlistener_case_name: str | None,
    *,
    message: str | None = None,
    chat_history: Iterable[ChatTurn] | None = None,
) -> CaseNameAssessment:
    """Build a case-name assessment with the canonical status message."""
    return CaseNameAssessment(
        citation_id=citation_id,
        status=status,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
        message=message or _STATUS_MESSAGES.get(status, "Case name has not been assessed."),
        chat_history=tuple(chat_history) if chat_history is not None else None,
    )


def assess_case_name_exact_match(
    *,
    citation_id: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str | None,
) -> CaseNameAssessment:
    """Short-circuit case-name assessment when exact string equality is enough."""
    if not extracted_case_name:
        return build_case_name_assessment(
            citation_id,
            CaseNameAssessmentStatus.NEEDS_ASSESSMENT,
            extracted_case_name,
            courtlistener_case_name,
            message="No case name was extracted; needs assessment against local context.",
        )
    if not courtlistener_case_name:
        return build_case_name_assessment(
            citation_id,
            CaseNameAssessmentStatus.NEEDS_ASSESSMENT,
            extracted_case_name,
            courtlistener_case_name,
            message="No CourtListener case name is available to compare against.",
        )
    if case_names_equivalent(extracted_case_name, courtlistener_case_name):
        return build_case_name_assessment(
            citation_id,
            CaseNameAssessmentStatus.EXACT_MATCH,
            extracted_case_name,
            courtlistener_case_name,
        )
    return build_case_name_assessment(
        citation_id,
        CaseNameAssessmentStatus.NEEDS_ASSESSMENT,
        extracted_case_name,
        courtlistener_case_name,
        message="Extracted case name differs from CourtListener and needs assessment.",
    )
