"""Case-name assessment helpers."""

import unicodedata

from mellea_lrc.assessment.types import CaseNameAssessment, CaseNameAssessmentStatus
from mellea_lrc.core.citations import FullCaseCitation

# Typographic variants that should not count as a difference between an extracted
# case name and the retrieved record (e.g. a curly apostrophe copied from a PDF
# versus the straight apostrophe CourtListener stores).
_TYPOGRAPHIC_TRANSLATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‛": "'",
        "′": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "−": "-",
    }
)


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


def assess_case_name_exact_match(
    *,
    citation_id: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str | None,
) -> CaseNameAssessment:
    """Short-circuit case-name assessment when exact string equality is enough.

    Any non-exact outcome is reported as ``NEEDS_ASSESSMENT`` so the caller routes
    it to the model-backed classification and local-context re-extraction. A
    missing extracted name is also routed there so re-extraction can try to
    recover the parties from local context.
    """
    if not extracted_case_name:
        return CaseNameAssessment(
            citation_id=citation_id,
            status=CaseNameAssessmentStatus.NEEDS_ASSESSMENT,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            message="No case name was extracted; needs assessment against local context.",
        )
    if not courtlistener_case_name:
        return CaseNameAssessment(
            citation_id=citation_id,
            status=CaseNameAssessmentStatus.NEEDS_ASSESSMENT,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            message="No CourtListener case name is available to compare against.",
        )
    if case_names_equivalent(extracted_case_name, courtlistener_case_name):
        return CaseNameAssessment(
            citation_id=citation_id,
            status=CaseNameAssessmentStatus.EXACT_MATCH,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            message="Extracted case name exactly matches CourtListener.",
        )
    return CaseNameAssessment(
        citation_id=citation_id,
        status=CaseNameAssessmentStatus.NEEDS_ASSESSMENT,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
        message="Extracted case name differs from CourtListener and needs assessment.",
    )
