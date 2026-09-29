"""One grounding rule for quotes copied from independent body evidence."""

from __future__ import annotations

from typing import TypeVar

from mellea_lrc.matching.fuzziness import FuzzinessOption
from mellea_lrc.matching.grounding import GroundedFragment, GroundingEvidence

T = TypeVar("T")
MIN_SIMILARITY = 98.0
_FUZZINESS = FuzzinessOption.edit_distance(similarity_percent=MIN_SIMILARITY, whitespace_relaxation=True)


def ground_body_fragment(evidence: GroundingEvidence[T], proposed: str) -> GroundedFragment[T] | None:
    """Apply edit and whitespace tolerance without accepting a short-token typo."""
    found = evidence.find_fragment(proposed, _FUZZINESS)
    return found if found is not None and found.similarity_percent >= MIN_SIMILARITY else None
