"""Source-window grounding shared by both docket lookup review contexts."""

from __future__ import annotations

from mellea_lrc.matching.fuzziness import FuzzinessOption
from mellea_lrc.matching.grounding import EvidenceCandidate, GroundingEvidence
from mellea_lrc.model.citations.docket_lookup import DocketLookupReviewDecision
from mellea_lrc.model.span import Span


class DocketReviewGrounding:
    """Ground docket, name, court and date proposals in their original windows."""

    number_window: str
    number_offset: int
    before_window: str
    before_offset: int
    after_window: str
    after_offset: int

    def grounded_corrections(self, decision: DocketLookupReviewDecision) -> dict[str, Span] | None:
        """Locate every proposed reading in its allowed filing window."""
        corrections: dict[str, Span] = {}
        windows = {
            "docket_number": (self.number_window, self.number_offset),
            "case_name": (self.before_window, self.before_offset),
            "court": (self.after_window, self.after_offset),
            "date": (self.after_window, self.after_offset),
        }
        for field, (window, offset) in windows.items():
            assessment = getattr(decision, field)
            if not assessment.propose_replacement:
                continue
            if assessment.quote is None:
                return None
            found = GroundingEvidence((EvidenceCandidate(window, offset),)).find_fragment(
                assessment.quote,
                FuzzinessOption.edit_distance(similarity_percent=90, whitespace_relaxation=True),
            )
            if found is None:
                return None
            corrections[field] = Span(offset + found.start, offset + found.end)
        return corrections
