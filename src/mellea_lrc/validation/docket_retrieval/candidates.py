"""Score docket shortlist candidates without deciding case identity."""

from __future__ import annotations

from mellea_lrc.matching.fuzziness import FuzzinessOption
from mellea_lrc.matching.grounding import EvidenceCandidate, GroundingEvidence

MINIMUM_SIMILARITY_PERCENT = 40.0
_SHORTLIST_FUZZINESS = FuzzinessOption.edit_distance(
    similarity_percent=MINIMUM_SIMILARITY_PERCENT, whitespace_relaxation=True
)
_SCORE_FUZZINESS = FuzzinessOption.edit_distance(similarity_percent=1.0, whitespace_relaxation=True)


def docket_number_similarity(source_number: str, candidate_number: str | None) -> float:
    """Use the shared fuzzy matcher, retaining scores for below-cutoff hits."""
    if not candidate_number:
        return 0.0
    evidence = GroundingEvidence((EvidenceCandidate(candidate_number.casefold(), None),))
    for policy in (_SHORTLIST_FUZZINESS, _SCORE_FUZZINESS):
        match = evidence.resolve(source_number.casefold(), policy)
        if match is not None:
            return match.similarity_percent
    return 0.0
