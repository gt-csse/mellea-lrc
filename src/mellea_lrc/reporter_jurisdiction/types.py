"""Immutable reporter-jurisdiction inference types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ReporterJurisdictionStatus(str, Enum):
    """Whether a reporter was present, recognized, and jurisdictionally useful."""

    MISSING_REPORTER = "missing_reporter"
    UNRECOGNIZED = "unrecognized"
    RECOGNIZED_WITHOUT_CONSTRAINT = "recognized_without_constraint"
    CONSTRAINED = "constrained"


class ReporterCoverage(str, Enum):
    """How completely the recorded constraints describe a reporter's scope."""

    UNKNOWN = "unknown"
    PARTIAL = "partial"
    EXHAUSTIVE = "exhaustive"


class CourtClass(str, Enum):
    """Court classes useful for retrieval without asserting an exact court."""

    FEDERAL_SUPREME = "federal_supreme"
    FEDERAL_APPELLATE = "federal_appellate"
    FEDERAL_DISTRICT = "federal_district"
    FEDERAL_BANKRUPTCY = "federal_bankruptcy"
    SPECIALIZED_FEDERAL = "specialized_federal"


@dataclass(frozen=True, slots=True)
class ReporterJurisdictionEvidence:
    """One provenance-bearing statement supporting an inference."""

    source: str
    statement: str

    def __post_init__(self) -> None:
        if not self.source or not self.statement:
            msg = "Reporter jurisdiction evidence requires source and statement"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ReporterJurisdictionInference:
    """Jurisdiction constraints supplied by a reporter designation.

    Exact-court inference is deliberately a projection of this representation:
    it exists only for an exhaustive singleton court set.
    """

    reporter: str | None
    status: ReporterJurisdictionStatus
    court_ids: tuple[str, ...] = ()
    court_classes: tuple[CourtClass, ...] = ()
    jurisdiction_ids: tuple[str, ...] = ()
    coverage: ReporterCoverage = ReporterCoverage.UNKNOWN
    evidence: tuple[ReporterJurisdictionEvidence, ...] = ()

    def __post_init__(self) -> None:
        if self.reporter is not None and not self.reporter.strip():
            msg = "Reporter must be None or a non-empty string"
            raise ValueError(msg)
        for label, values in (
            ("court_ids", self.court_ids),
            ("court_classes", self.court_classes),
            ("jurisdiction_ids", self.jurisdiction_ids),
        ):
            if len(values) != len(set(values)):
                msg = f"Reporter jurisdiction {label} must be unique"
                raise ValueError(msg)

        constrained = bool(self.court_ids or self.court_classes or self.jurisdiction_ids)
        if self.status is ReporterJurisdictionStatus.MISSING_REPORTER and self.reporter is not None:
            msg = "Missing-reporter inference cannot carry a reporter"
            raise ValueError(msg)
        if self.status in {
            ReporterJurisdictionStatus.MISSING_REPORTER,
            ReporterJurisdictionStatus.UNRECOGNIZED,
            ReporterJurisdictionStatus.RECOGNIZED_WITHOUT_CONSTRAINT,
        } and constrained:
            msg = f"{self.status.value} inference cannot carry jurisdiction constraints"
            raise ValueError(msg)
        if self.status is ReporterJurisdictionStatus.CONSTRAINED and not constrained:
            msg = "Constrained reporter inference requires at least one constraint"
            raise ValueError(msg)
        if self.coverage is ReporterCoverage.EXHAUSTIVE and not constrained:
            msg = "Exhaustive reporter inference requires jurisdiction constraints"
            raise ValueError(msg)
        if self.status is ReporterJurisdictionStatus.CONSTRAINED and not self.evidence:
            msg = "Constrained reporter inference requires provenance evidence"
            raise ValueError(msg)

    @property
    def exact_court_id(self) -> str | None:
        """Return the court only for an exhaustive singleton court mapping."""
        if self.coverage is ReporterCoverage.EXHAUSTIVE and len(self.court_ids) == 1:
            return self.court_ids[0]
        return None


class ReporterJurisdictionCompatibilityStatus(str, Enum):
    """Relationship between reporter constraints and explicit candidate metadata."""

    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class ReporterJurisdictionCompatibility:
    """Non-mutating comparison of one inference with one candidate court."""

    status: ReporterJurisdictionCompatibilityStatus
    candidate_court_id: str | None
    candidate_court_class: CourtClass | None
    candidate_jurisdiction_id: str | None
    message: str

