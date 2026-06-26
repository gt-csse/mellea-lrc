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

    First pass (before re-extraction): ``exact_match``, ``semantic_match``, or
    route to re-extraction via internal ``needs_assessment``.

    After re-extraction: ``exact_match``, ``semantic_match``, ``different_case``,
    ``irregular_form``, or ``reextraction_fail``.
    """

    EXACT_MATCH = "exact_match"
    SEMANTIC_MATCH = "semantic_match"
    DIFFERENT_CASE = "different_case"
    IRREGULAR_FORM = "irregular_form"
    REEXTRACTION_FAIL = "reextraction_fail"
    NEEDS_ASSESSMENT = "needs_assessment"


class YearAssessmentStatus(str, Enum):
    """Canonical outcomes for deterministic year assessment."""

    EXACT_MATCH = "exact_match"
    MISMATCH = "mismatch"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class CaseNameAssessment:
    """Assessment result for one extracted case name."""

    citation_id: str
    status: CaseNameAssessmentStatus
    extracted_case_name: str | None
    courtlistener_case_name: str | None
    message: str
    chat_history: list[dict[str, str]] | None = None


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
    case_assess: CaseNameAssessment
    year_assess: YearAssessment


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
    """LLM-proposed corrected case name copied from local context."""

    case_name: str | None = None

    @property
    def extracted_case_name(self) -> str | None:
        """Alias used by reassessment and serialization."""
        return self.case_name

    def valid(self, document_context: str) -> bool:
        """Require the proposed case name to be present in the original context."""
        from mellea_lrc.assessment.grounding.proposal import is_in_context

        return bool(self.case_name) and is_in_context(self.case_name, document_context)


@dataclass(frozen=True, slots=True)
class ModifiedExtractedCitation:
    """A grounded modified extraction bound to document-local citation identity."""

    citation_id: str = ""
    span: Span | None = None
    matched_text: str | None = None
    case_name: str | None = None

    @property
    def extracted_case_name(self) -> str | None:
        """Alias used by reassessment and serialization."""
        return self.case_name

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
            matched_text=proposal.case_name,
            case_name=proposal.case_name,
        )


@dataclass(frozen=True, slots=True)
class CaseNameAssessmentRun:
    """Case-name assessment plus any modified extraction history it produced."""

    assessment: CaseNameAssessment
    modified_citation: ModifiedExtractedCitationProposal | None = None
    reassessment: CaseNameAssessment | None = None
