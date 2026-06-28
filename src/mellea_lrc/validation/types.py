"""Validation result types."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, TypeAlias

from mellea_lrc.core.immutable import ExtraData
from mellea_lrc.courtlistener.types import CitationMatch, ValidationFailureDetail
from mellea_lrc.extraction.types import ExtractedDocument

ValidationClientMode: TypeAlias = Literal["deployed", "sdk", "custom"]


@dataclass(frozen=True, slots=True)
class ValidationMetadata:
    """Provenance for the validation stage."""

    client_mode: ValidationClientMode
    source: str


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
    lookup_status: int | None = None
    lookup_cache: str | None = None
    lookup_key: str | None = None
    error_message: str | None = None
    failure_detail: ValidationFailureDetail | None = None
    matches: tuple[CitationMatch, ...] = ()
    extra_data: ExtraData = field(default_factory=ExtraData)

    @property
    def case_names(self) -> tuple[str, ...]:
        """Return non-empty candidate case names."""
        return tuple(item.case_name for item in self.matches if item.case_name)


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidatedDocument(ExtractedDocument):
    """An extracted document with one validation outcome per citation."""

    validations: tuple[CitationValidation, ...]
    validation_metadata: ValidationMetadata

    @property
    def found(self) -> tuple[CitationValidation, ...]:
        """Return citations found by the validation source."""
        return tuple(item for item in self.validations if item.status == ValidationStatus.FOUND)

    def __post_init__(self) -> None:
        ExtractedDocument.__post_init__(self)
        citation_ids = {item.citation_id for item in self.citations}
        validation_ids = [item.citation_id for item in self.validations]
        if any(not validation_id for validation_id in validation_ids):
            msg = "Citation validation identifiers must not be empty"
            raise ValueError(msg)
        if len(validation_ids) != len(set(validation_ids)):
            msg = "Citation validation identifiers must be unique within a document"
            raise ValueError(msg)
        if set(validation_ids) != citation_ids:
            msg = "Citation validation identifiers must exactly match extracted citation identifiers"
            raise ValueError(msg)
