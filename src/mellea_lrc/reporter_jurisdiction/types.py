"""Immutable reporter-jurisdiction inference types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.taxonomy import CLCourtTaxonomy


class ReporterJurisdictionStatus(str, Enum):
    """Classification of a reporter string against the curated registry.

    ``VALID_REPORTER`` and ``EXHAUSTIVE_REPORTER`` form a hierarchy:
    every ``EXHAUSTIVE_REPORTER`` is also a ``VALID_REPORTER``.
    Downstream guards that only need to know "is this a known reporter?"
    should accept both statuses.
    """

    MISSING_REPORTER = "missing_reporter"
    UNRECOGNIZED = "unrecognized"
    VALID_REPORTER = "valid_reporter"
    EXHAUSTIVE_REPORTER = "exhaustive_reporter"


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

    ``exact_court_id`` is a projection available only for
    ``EXHAUSTIVE_REPORTER`` status, where the reporter maps to exactly
    one CourtListener court.
    """

    reporter: str | None
    status: ReporterJurisdictionStatus
    court_ids: tuple[str, ...] = ()
    evidence: tuple[ReporterJurisdictionEvidence, ...] = ()
    cl_court_taxonomy: "CLCourtTaxonomy | None" = None

    def __post_init__(self) -> None:
        if self.reporter is not None and not self.reporter.strip():
            msg = "Reporter must be None or a non-empty string"
            raise ValueError(msg)
        if len(self.court_ids) != len(set(self.court_ids)):
            msg = "Reporter jurisdiction court_ids must be unique"
            raise ValueError(msg)

        is_terminal = self.status in {
            ReporterJurisdictionStatus.MISSING_REPORTER,
            ReporterJurisdictionStatus.UNRECOGNIZED,
            ReporterJurisdictionStatus.VALID_REPORTER,
        }
        if self.status is ReporterJurisdictionStatus.MISSING_REPORTER and self.reporter is not None:
            msg = "Missing-reporter inference cannot carry a reporter"
            raise ValueError(msg)
        if is_terminal and self.court_ids:
            msg = f"{self.status.value} inference cannot carry court_ids"
            raise ValueError(msg)
        if self.status is ReporterJurisdictionStatus.EXHAUSTIVE_REPORTER:
            if len(self.court_ids) != 1:
                msg = "Exhaustive reporter inference requires exactly one court_id"
                raise ValueError(msg)
            if not self.evidence:
                msg = "Exhaustive reporter inference requires provenance evidence"
                raise ValueError(msg)

    @property
    def exact_court_id(self) -> str | None:
        """Return the court ID for an exhaustive reporter, otherwise None."""
        if self.status is ReporterJurisdictionStatus.EXHAUSTIVE_REPORTER:
            return self.court_ids[0]
        return None
