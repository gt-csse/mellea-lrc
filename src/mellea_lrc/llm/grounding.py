"""Reusable deterministic grounding for values proposed by a language model.

A model's text never enters a record directly.  It must resolve to one of the
source candidates supplied by the caller, whose value normally carries the
source span or canonical text.  Matching starts exact and may add whitespace
and edit-distance fallbacks through an explicit policy.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import floor
from typing import Generic, TypeVar

from rapidfuzz.distance import Levenshtein

from mellea_lrc.core.fuzziness import FuzzinessOption, FuzzinessType

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class EvidenceCandidate(Generic[T]):
    """One canonical source value that a model proposal may resolve to."""

    text: str
    value: T


@dataclass(frozen=True, slots=True)
class GroundingMatch(Generic[T]):
    """A proposal resolved to one source candidate by one explicit stage."""

    candidate: EvidenceCandidate[T]
    match_type: FuzzinessType
    edits: int
    similarity_percent: float


@dataclass(frozen=True, slots=True)
class GroundingEvidence(Generic[T]):
    """A stable, ordered collection of source candidates for one grounding decision."""

    candidates: tuple[EvidenceCandidate[T], ...]

    def __init__(self, candidates: Iterable[EvidenceCandidate[T]]) -> None:
        object.__setattr__(self, "candidates", tuple(candidates))

    def fuzzy_match(
        self,
        proposed: str,
        fuzziness: FuzzinessOption,
    ) -> tuple[GroundingMatch[T], ...]:
        """Return every source candidate that satisfies the selected policy."""
        return fuzzy_match(proposed, self.candidates, fuzziness)

    def resolve_by_condition(
        self,
        condition: Callable[[EvidenceCandidate[T]], bool],
    ) -> EvidenceCandidate[T] | None:
        """Return the first source candidate satisfying a caller-owned condition."""
        return next((candidate for candidate in self.candidates if condition(candidate)), None)

    def resolve(self, proposed: str, fuzziness: FuzzinessOption) -> GroundingMatch[T] | None:
        """Return the first source-grounded match in the evidence's stable order."""
        return next(iter(self.fuzzy_match(proposed, fuzziness)), None)


_WHITESPACE = re.compile(r"\s+")


def fuzzy_match(
    proposed: str,
    candidates: Iterable[EvidenceCandidate[T]],
    fuzziness: FuzzinessOption,
) -> tuple[GroundingMatch[T], ...]:
    """Match a model-proposed value against canonical source candidates.

    Candidates retain their input order.  Callers that need a different tie
    policy can inspect every returned match or use ``resolve_by_condition``.
    """
    return tuple(
        match for candidate in candidates if (match := _match(proposed, candidate, fuzziness)) is not None
    )


def _match(
    proposed: str,
    candidate: EvidenceCandidate[T],
    fuzziness: FuzzinessOption,
) -> GroundingMatch[T] | None:
    if proposed == candidate.text:
        return GroundingMatch(candidate, FuzzinessType.PERFECT_MATCH, 0, 100.0)
    if fuzziness.types == frozenset({FuzzinessType.PERFECT_MATCH}):
        return None

    normalized_proposed = _normalize_whitespace(proposed)
    normalized_candidate = _normalize_whitespace(candidate.text)
    if FuzzinessType.WHITESPACE_RELAXATION in fuzziness.types and normalized_proposed == normalized_candidate:
        return GroundingMatch(candidate, FuzzinessType.WHITESPACE_RELAXATION, 0, 100.0)

    if FuzzinessType.EDIT_DISTANCE not in fuzziness.types:
        return None
    compared_proposed = (
        normalized_proposed if FuzzinessType.WHITESPACE_RELAXATION in fuzziness.types else proposed
    )
    compared_candidate = (
        normalized_candidate if FuzzinessType.WHITESPACE_RELAXATION in fuzziness.types else candidate.text
    )
    edits = Levenshtein.distance(compared_proposed, compared_candidate)
    maximum_edits = _maximum_edits(compared_proposed, compared_candidate, fuzziness)
    if edits > maximum_edits:
        return None
    length = max(len(compared_proposed), len(compared_candidate))
    similarity = 100.0 if length == 0 else 100.0 * (1 - edits / length)
    return GroundingMatch(candidate, FuzzinessType.EDIT_DISTANCE, edits, similarity)


def _maximum_edits(proposed: str, candidate: str, fuzziness: FuzzinessOption) -> int:
    if fuzziness.maximum_edits is not None:
        return fuzziness.maximum_edits
    assert fuzziness.similarity_percent is not None
    if fuzziness.similarity_percent == 100:
        return 0
    error_rate = 1 - fuzziness.similarity_percent / 100
    return max(1, floor(max(len(proposed), len(candidate)) * error_rate))


def _normalize_whitespace(value: str) -> str:
    """Erase layout-only whitespace for the project's relaxed-string policy.

    ``fuzzy_literal(..., whitespace_relaxation())`` accepts zero or more
    whitespace characters at token joins. Grounding must mean the same thing:
    ``1: 24-cv`` and ``1:24-cv`` differ only in layout, while changing a
    punctuation mark or digit remains a substantive mismatch.
    """
    return _WHITESPACE.sub("", value)
