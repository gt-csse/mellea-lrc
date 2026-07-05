"""Jurisdiction Leads core types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.taxonomy import CLCourtTaxonomy


class ReporterLeadStatus(str, Enum):
    """Classification of a reporter string."""
    MISSING_REPORTER = "missing_reporter"
    UNRECOGNIZED = "unrecognized"
    RECOGNIZED = "recognized"


class CourtLeadStatus(str, Enum):
    """Classification of an extracted court string."""
    MISSING_COURT = "missing_court"
    UNRECOGNIZED = "unrecognized"
    RESOLVED = "resolved"


@dataclass(frozen=True, slots=True)
class ReporterLead:
    """Reporter-based jurisdiction lead."""
    reporter: str | None
    status: ReporterLeadStatus
    mlz_jurisdictions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CourtLead:
    """Court-based jurisdiction lead."""
    extracted_court: str | None
    status: CourtLeadStatus
    cl_court_taxonomy: CLCourtTaxonomy | None


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
class JurisdictionInference:
    """Combined jurisdiction leads for a citation."""
    reporter_lead: ReporterLead
    court_lead: CourtLead
