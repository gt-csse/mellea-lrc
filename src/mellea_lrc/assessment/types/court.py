"""Citation-court assessment types."""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, TypeAlias


class CourtAssessmentStatus(str, Enum):
    """Canonical outcomes for deterministic court assessment."""

    EXACT_MATCH = "exact_match"
    MISMATCH = "mismatch"
    MISSING = "missing"


class CourtFollowupStatus(str, Enum):
    """Execution outcome after the initial court assessment."""

    NOT_REQUIRED = "not_required"
    INFERRED_FROM_REPORTER = "inferred_from_reporter"


@dataclass(frozen=True, slots=True)
class CourtAssessment:
    """Substantive comparison of a citation court slug and the CourtListener court."""

    status: CourtAssessmentStatus
    extracted_court: str | None
    courtlistener_court_id: str | None
    message: str


@dataclass(frozen=True, slots=True)
class CourtFollowupNotRequired:
    """The initial court assessment required no reporter inference."""

    status: ClassVar[CourtFollowupStatus] = CourtFollowupStatus.NOT_REQUIRED


@dataclass(frozen=True, slots=True)
class CourtInferredFromReporter:
    """Reporter inference filled a missing extracted court before reassessment."""

    status: ClassVar[CourtFollowupStatus] = CourtFollowupStatus.INFERRED_FROM_REPORTER
    reporter: str | None
    citation_court_before: str | None
    result: CourtAssessment


CourtFollowup: TypeAlias = CourtFollowupNotRequired | CourtInferredFromReporter


@dataclass(frozen=True, slots=True)
class CourtAssessmentRun:
    """Initial court assessment and its field-local follow-up outcome."""

    initial: CourtAssessment
    followup: CourtFollowup

    @property
    def final(self) -> CourtAssessment:
        """The terminal court verdict after any reporter-inference follow-up.

        When reporter inference ran, its reassessment is the conclusion the
        citation should be read by; otherwise the initial comparison stands.
        This is the single value downstream consumers (counts, tags) should
        report, mirroring how ``YearAssessment`` and the reassessed
        ``CaseNameAssessment`` expose a terminal per-field verdict.
        """
        if isinstance(self.followup, CourtInferredFromReporter):
            return self.followup.result
        return self.initial

    def __post_init__(self) -> None:
        if isinstance(self.followup, CourtInferredFromReporter):
            if self.initial.status != CourtAssessmentStatus.MISSING:
                msg = "Reporter inference requires an initial missing court assessment"
                raise ValueError(msg)
            if self.followup.result.status not in {
                CourtAssessmentStatus.EXACT_MATCH,
                CourtAssessmentStatus.MISMATCH,
                CourtAssessmentStatus.MISSING,
            }:
                msg = "Reporter inference follow-up must contain a terminal court assessment"
                raise ValueError(msg)
