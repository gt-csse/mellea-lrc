"""Jurisdiction Leads core types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from mellea_lrc.extraction.types import ExtractedDocument

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.taxonomy import CourtsDBClassification
    from mellea_lrc.core.citations import Reporter


class ReporterInferenceStatus(str, Enum):
    """Classification of a reporter inference."""

    UNSUPPORTED = "unsupported"
    MISSING_REPORTER = "missing_reporter"
    UNRECOGNIZED = "unrecognized"
    RECOGNIZED = "recognized"


class CourtInferenceStatus(str, Enum):
    """Classification of an extracted court string."""

    UNSUPPORTED = "unsupported"
    MISSING_COURT = "missing_court"
    UNRECOGNIZED = "unrecognized"
    RESOLVED = "resolved"


@dataclass(frozen=True, slots=True)
class ReporterInference:
    """Reporter-based jurisdiction inference."""

    reporter: Reporter | None
    status: ReporterInferenceStatus
    mlz_jurisdictions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CourtInference:
    """Court-based jurisdiction inference."""

    extracted_court: str | None
    status: CourtInferenceStatus
    courts_db_classification: CourtsDBClassification | None


class TranslationLayerStatus(str, Enum):
    """Classification of the translation layer output."""

    UNAVAILABLE = "unavailable"
    AMBIGUOUS = "ambiguous"
    RESOLVED = "resolved"


@dataclass(frozen=True, slots=True)
class TranslationLayerResult:
    """The result of triangulating MLZ and Court Leads."""

    status: TranslationLayerStatus
    translated_court_id: str | None


@dataclass(frozen=True, slots=True)
class Jurisdiction:
    """Combined jurisdiction leads for a citation."""

    reporter_inference: ReporterInference
    court_inference: CourtInference


@dataclass(frozen=True, slots=True, kw_only=True)
class InferredDocument(ExtractedDocument):
    """An extracted document with jurisdiction inference per citation."""

    jurisdictions: tuple[Jurisdiction, ...]

    def __post_init__(self) -> None:
        ExtractedDocument.__post_init__(self)
        if len(self.jurisdictions) != len(self.citations):
            msg = (
                f"Inferences count {len(self.jurisdictions)} must match "
                f"citations count {len(self.citations)}"
            )
            raise ValueError(msg)
