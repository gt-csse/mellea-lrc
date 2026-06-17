"""Validation result types."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias

from mellea_lrc.courtlistener.types import JsonObject

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
    """Validation results for one extracted document."""

    validations: tuple[CitationValidation, ...]

    @property
    def found(self) -> tuple[CitationValidation, ...]:
        """Return citations found by the validation source."""
        return tuple(item for item in self.validations if item.status == ValidationStatus.FOUND)
