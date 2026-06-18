"""Validation result types."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias

from mellea_lrc.courtlistener.types import JsonObject
from mellea_lrc.extraction.types import ExtractedCitation
from mellea_lrc.preprocessing.types import PreprocessedDocument

ValidationClientMode: TypeAlias = Literal["deployed", "sdk", "custom"]


class ValidationStatus(str, Enum):
    """Canonical validation outcomes for citation existence checks."""

    FOUND = "found"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"
    INVALID = "invalid"
    THROTTLED = "throttled"
    LOOKUP_FAILED = "lookup_failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class CitationValidation:
    """Validation result for one extracted citation."""

    citation_id: str
    locator: str | None
    status: ValidationStatus
    source: str
    message: str
    case_names: tuple[str, ...] = ()
    lookup_status: int | None = None
    lookup_cache: str | None = None
    lookup_key: str | None = None
    error_message: str | None = None
    limit_detail: JsonObject | None = None
    clusters: tuple[JsonObject, ...] = ()


@dataclass(frozen=True, slots=True)
class DocumentValidation:
    """Validated citations for one preprocessed document."""

    preprocessed: PreprocessedDocument
    citations: tuple[ExtractedCitation, ...]
    validations: tuple[CitationValidation, ...]

    @property
    def text(self) -> str:
        """Text that was validated."""
        return self.preprocessed.text

    @property
    def source_path(self) -> str | None:
        """Original source path, when known."""
        return self.preprocessed.metadata.source_path

    @property
    def found(self) -> tuple[CitationValidation, ...]:
        """Return citations found by the validation source."""
        return tuple(item for item in self.validations if item.status == ValidationStatus.FOUND)
