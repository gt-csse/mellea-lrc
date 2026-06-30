"""Citation-level assessment types."""

from dataclasses import dataclass

from mellea_lrc.assessment.types.case_name import CaseNameAssessmentRun
from mellea_lrc.assessment.types.court import CourtAssessment
from mellea_lrc.assessment.types.year import YearAssessment


@dataclass(frozen=True, slots=True)
class CitationAssessmentResult:
    """Completed field assessments for one citation."""

    case_name: CaseNameAssessmentRun
    court: CourtAssessment
    year: YearAssessment
