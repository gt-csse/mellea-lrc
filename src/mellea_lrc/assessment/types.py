"""Assessment types for LLM-assisted citation checks."""

from dataclasses import dataclass
from enum import Enum

from mellea_lrc.extraction.types import ExtractedCitation
from mellea_lrc.preprocessing.types import PreprocessedDocument
from mellea_lrc.validation.types import CitationValidation


class CaseNameAssessmentStatus(str, Enum):
    """Canonical outcomes for case-name assessment."""

    EXACT_MATCH = "exact_match"
    SEMANTIC_MATCH = "semantic_match"
    NEEDS_SEMANTIC_ASSESSMENT = "needs_semantic_assessment"
    EXTRACTION_ERROR = "extraction_error"


class YearAssessmentStatus(str, Enum):
    """Canonical outcomes for deterministic year assessment."""

    EXACT_MATCH = "exact_match"
    MISMATCH = "mismatch"
    MISSING = "missing"


class CitationAssessmentStatus(str, Enum):
    """Canonical roll-up outcomes for one citation assessment."""

    EXACT_MATCH = "exact_match"
    SEMANTIC_MATCH = "semantic_match"
    NEEDS_SEMANTIC_ASSESSMENT = "needs_semantic_assessment"
    EXTRACTION_ERROR = "extraction_error"


@dataclass(frozen=True, slots=True)
class CaseNameAssessment:
    """Assessment result for one extracted case name."""

    citation_id: str
    status: CaseNameAssessmentStatus
    extracted_case_name: str | None
    courtlistener_case_name: str | None
    message: str
    modified_extracted_case_name: str | None = None
    modified_match_status: str | None = None


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
        """Roll up specific assessment statuses for this citation."""
        if self.case_assess is None:
            return CitationAssessmentStatus.EXTRACTION_ERROR
        if self.case_assess.status == CaseNameAssessmentStatus.EXTRACTION_ERROR:
            return CitationAssessmentStatus.EXTRACTION_ERROR
        if self.year_assess is not None and self.year_assess.status != YearAssessmentStatus.EXACT_MATCH:
            return CitationAssessmentStatus.EXTRACTION_ERROR
        if self.case_assess.status == CaseNameAssessmentStatus.SEMANTIC_MATCH:
            return CitationAssessmentStatus.SEMANTIC_MATCH
        if self.case_assess.status == CaseNameAssessmentStatus.NEEDS_SEMANTIC_ASSESSMENT:
            return CitationAssessmentStatus.NEEDS_SEMANTIC_ASSESSMENT
        return CitationAssessmentStatus.EXACT_MATCH

    @property
    def message(self) -> str:
        """Roll up a display message for this citation."""
        if self.year_assess is not None and self.year_assess.status != YearAssessmentStatus.EXACT_MATCH:
            return self.year_assess.message
        if self.status == CitationAssessmentStatus.EXACT_MATCH:
            return "Assessed bibliographic fields match CourtListener."
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

    @property
    def text(self) -> str:
        """Text that was assessed."""
        return self.preprocessed.text

    @property
    def source_path(self) -> str | None:
        """Original source path, when known."""
        return self.preprocessed.metadata.source_path


@dataclass(frozen=True, slots=True)
class ModifiedExtractedCitation:
    """A grounded correction to the extracted case-name fields."""

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


def is_in_context(value: str, document_context: str) -> bool:
    """Return whether a proposed value is grounded in the source context."""
    return _normalize_whitespace(value) in _normalize_whitespace(document_context)


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())
