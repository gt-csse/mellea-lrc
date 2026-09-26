"""Source-grounded field proposals shared by reporter model-review stages."""

from __future__ import annotations

from mellea_lrc.llm.fuzziness import FuzzinessOption
from mellea_lrc.llm.grounding import EvidenceCandidate, GroundingEvidence
from mellea_lrc.model.citations.judgments import MatchResult
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterAmbiguousReviewDecision,
    ReporterUniqueReviewDecision,
)
from mellea_lrc.model.span import Span

ReporterReviewDecision = ReporterUniqueReviewDecision | ReporterAmbiguousReviewDecision


class ReporterReviewGrounding:
    """Methods for contexts with bounded source windows and current readings."""

    case_name_window: str
    case_name_offset: int
    following_window: str
    following_offset: int
    has_case_name: bool
    has_court: bool
    has_date: bool

    def ground(self, field: str, quote: str) -> Span | None:
        """Resolve a proposed reading to exact bytes in its allowed window."""
        if field == "case_name":
            window, offset = self.case_name_window, self.case_name_offset
        elif field in {"court", "date"}:
            window, offset = self.following_window, self.following_offset
        else:
            raise ValueError(f"Unknown reporter review field: {field}")
        found = GroundingEvidence((EvidenceCandidate(window, offset),)).find_fragment(
            quote,
            FuzzinessOption.edit_distance(similarity_percent=90, whitespace_relaxation=True),
        )
        if found is None:
            return None
        return Span(offset + found.start, offset + found.end)

    def grounded_corrections(self, decision: ReporterReviewDecision) -> dict[str, Span] | None:
        """Validate intent and ground every proposed quote before an update."""
        corrections: dict[str, Span] = {}
        for field in ("case_name", "court", "date"):
            assessment = getattr(decision, field)
            quote = assessment.quote
            if not assessment.propose_replacement:
                if quote is not None:
                    return None
                continue
            if quote is None or not quote.strip():
                return None
            span = self.ground(field, quote)
            if span is None:
                return None
            corrections[field] = span
        return corrections

    def assessment_error(self, decision: ReporterReviewDecision) -> str | None:
        """A match or mismatch needs a filing reading, existing or proposed."""
        has_name = self.has_case_name or decision.case_name.propose_replacement
        if has_name and decision.case_name.normalized is None:
            return "case_name has a grounded reading; supply its normalized name"
        if not has_name and decision.case_name.normalized is not None:
            return "case_name normalization needs an existing or proposed grounded reading"
        for field in ("case_name", "court", "date"):
            assessment = getattr(decision, field)
            if not getattr(self, f"has_{field}") and not assessment.propose_replacement:
                if assessment.result is not MatchResult.UNDETERMINED:
                    return f"{field} has no filing reading; use undetermined or quote one from the filing"
        return None
