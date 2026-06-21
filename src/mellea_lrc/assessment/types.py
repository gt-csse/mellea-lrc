"""Assessment types for LLM-assisted citation checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mellea_lrc.core.spans import Span
    from mellea_lrc.extraction.types import ExtractedCitation
    from mellea_lrc.preprocessing.types import PreprocessedDocument
    from mellea_lrc.validation.types import CitationValidation


class CaseNameAssessmentStatus(str, Enum):
    """Canonical outcomes for case-name assessment.

    ``EXACT_MATCH`` and ``NEEDS_ASSESSMENT`` are deterministic, internal states.
    ``MATCH`` / ``DIFFERENT_CASE`` / ``IRREGULAR_FORM`` are the verdicts the model
    returns, so their string values double as the prompt vocabulary:

    - ``match`` — same case as the retrieved record; abbreviation, party
      shortening, and omitted institutional suffixes are acceptable.
    - ``different_case`` — the extracted name denotes a different, unrelated case
      than the retrieved record (often a retrieval/citation problem, not the
      extractor's fault).
    - ``irregular_form`` — the same case, but written with unusual omission or
      shorthand (a likely mis-extraction worth re-checking against local context).
    """

    EXACT_MATCH = "exact_match"
    MATCH = "match"
    DIFFERENT_CASE = "different_case"
    IRREGULAR_FORM = "irregular_form"
    NEEDS_ASSESSMENT = "needs_assessment"


class YearAssessmentStatus(str, Enum):
    """Canonical outcomes for deterministic year assessment."""

    EXACT_MATCH = "exact_match"
    MISMATCH = "mismatch"
    MISSING = "missing"


class CitationAssessmentStatus(str, Enum):
    """Canonical roll-up outcomes for one citation assessment.

    Mirrors :class:`CaseNameAssessmentStatus`; the citation status currently rolls
    up the case-name verdict. Year is assessed independently and surfaced on its
    own field rather than folded into this status.
    """

    EXACT_MATCH = "exact_match"
    MATCH = "match"
    DIFFERENT_CASE = "different_case"
    IRREGULAR_FORM = "irregular_form"
    NEEDS_ASSESSMENT = "needs_assessment"


@dataclass(frozen=True, slots=True)
class CaseNameAssessment:
    """Assessment result for one extracted case name."""

    citation_id: str
    status: CaseNameAssessmentStatus
    extracted_case_name: str | None
    courtlistener_case_name: str | None
    message: str


@dataclass(frozen=True, slots=True)
class YearAssessment:
    """Assessment result for citation year."""

    citation_id: str
    status: YearAssessmentStatus
    extracted_year: str | None
    courtlistener_year: str | None
    message: str


@dataclass(frozen=True, slots=True)
class CitationAssessment:
    """Assessment result for one validated citation."""

    citation_id: str
    case_assess: CaseNameAssessment | None = None
    year_assess: YearAssessment | None = None

    @property
    def status(self) -> CitationAssessmentStatus:
        """Roll up the case-name verdict for this citation.

        Year is assessed separately (see :attr:`year_assess`) and is not folded
        into this status; a year mismatch is surfaced on the year field instead.
        """
        if self.case_assess is None:
            return CitationAssessmentStatus.NEEDS_ASSESSMENT
        return CitationAssessmentStatus(self.case_assess.status.value)

    @property
    def message(self) -> str:
        """Roll up a display message for this citation."""
        if self.case_assess is not None:
            return self.case_assess.message
        return "Citation assessment is missing case-name assessment."


@dataclass(frozen=True, slots=True)
class DocumentAssessment:
    """Assessed citations for one validated document."""

    preprocessed: PreprocessedDocument
    citations: tuple[ExtractedCitation, ...]
    validations: tuple[CitationValidation, ...]
    assessments: tuple[CitationAssessment, ...]
    modified_citations: tuple[ModifiedExtractedCitation, ...] = ()
    reassessments: tuple[CitationAssessment, ...] = ()

    @property
    def text(self) -> str:
        """Text that was assessed."""
        return self.preprocessed.text

    @property
    def source_path(self) -> str | None:
        """Original source path, when known."""
        return self.preprocessed.metadata.source_path


@dataclass(frozen=True, slots=True)
class ModifiedExtractedCitationProposal:
    """LLM-proposed correction to extracted case-name fields."""

    plaintiff: str | None = None
    defendant: str | None = None
    case_name: str | None = None

    @property
    def extracted_case_name(self) -> str | None:
        """Return the corrected case name used for follow-up assessment."""
        if self.case_name:
            return self.case_name
        if self.plaintiff and self.defendant:
            return f"{self.plaintiff} v. {self.defendant}"
        return self.plaintiff or self.defendant

    def valid(self, document_context: str) -> bool:
        """Require every proposed string field to be present in the original context."""
        values = tuple(value for value in (self.plaintiff, self.defendant, self.case_name) if value)
        return bool(values) and all(is_in_context(value, document_context) for value in values)


@dataclass(frozen=True, slots=True)
class ModifiedExtractedCitation:
    """A grounded modified extraction bound to document-local citation identity."""

    citation_id: str = ""
    span: Span | None = None
    matched_text: str | None = None
    plaintiff: str | None = None
    defendant: str | None = None
    case_name: str | None = None

    @property
    def extracted_case_name(self) -> str | None:
        """Return the corrected case name used for follow-up assessment."""
        if self.case_name:
            return self.case_name
        if self.plaintiff and self.defendant:
            return f"{self.plaintiff} v. {self.defendant}"
        return self.plaintiff or self.defendant

    @classmethod
    def from_proposal(
        cls,
        proposal: ModifiedExtractedCitationProposal,
        *,
        citation_id: str,
        span: Span | None,
    ) -> ModifiedExtractedCitation:
        """Bind an LLM-proposed extraction to document-local citation identity."""
        return cls(
            citation_id=citation_id,
            span=span,
            matched_text=proposal.extracted_case_name,
            plaintiff=proposal.plaintiff,
            defendant=proposal.defendant,
            case_name=proposal.case_name,
        )


def is_in_context(value: str, document_context: str) -> bool:
    """Return whether a proposed value is grounded in the source context."""
    return _normalize_whitespace(value) in _normalize_whitespace(document_context)


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


@dataclass(frozen=True, slots=True)
class CaseNameAssessmentRun:
    """Case-name assessment plus any modified extraction history it produced."""

    assessment: CaseNameAssessment
    modified_citation: ModifiedExtractedCitationProposal | None = None
    reassessment: CaseNameAssessment | None = None
