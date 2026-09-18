"""Reusable deterministic grounding for values proposed by a language model.

A model's text never enters a record directly.  It must resolve to one of the
source candidates supplied by the caller, whose value normally carries the
source span or canonical text.  Matching starts exact and may add whitespace
and edit-distance fallbacks through an explicit policy.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from math import floor
from typing import Generic, TypeVar

from rapidfuzz.distance import Levenshtein

T = TypeVar("T")


class FuzzinessType(str, Enum):
    """One permitted source-to-proposal comparison stage."""

    PERFECT_MATCH = "perfect_match"
    WHITESPACE_RELAXATION = "whitespace_relaxation"
    EDIT_DISTANCE = "edit_distance"


@dataclass(frozen=True, slots=True)
class FuzzinessOption:
    """An ordered, composable grounding policy.

    ``PERFECT_MATCH`` is an exclusive policy: it means no fallback is allowed.
    Whitespace relaxation and edit distance compose, with exact comparison
    always attempted first.  Edit distance runs on whitespace-normalized text
    whenever whitespace relaxation is enabled, so harmless layout variation
    does not spend the edit budget.
    """

    types: frozenset[FuzzinessType] = field(default_factory=lambda: frozenset({FuzzinessType.PERFECT_MATCH}))
    similarity_percent: float | None = None
    maximum_edits: int | None = None

    def __post_init__(self) -> None:
        normalized = frozenset(self.types)
        object.__setattr__(self, "types", normalized)
        if not normalized:
            msg = "A fuzziness policy needs at least one match type"
            raise ValueError(msg)
        if FuzzinessType.PERFECT_MATCH in normalized and len(normalized) != 1:
            msg = "PERFECT_MATCH cannot coexist with whitespace relaxation or edit distance"
            raise ValueError(msg)
        has_edit_distance = FuzzinessType.EDIT_DISTANCE in normalized
        if not has_edit_distance and (self.similarity_percent is not None or self.maximum_edits is not None):
            msg = "An edit-distance threshold requires EDIT_DISTANCE"
            raise ValueError(msg)
        if has_edit_distance and (self.similarity_percent is None) == (self.maximum_edits is None):
            msg = "EDIT_DISTANCE requires exactly one of similarity_percent or maximum_edits"
            raise ValueError(msg)
        if self.similarity_percent is not None and not 0 < self.similarity_percent <= 100:
            msg = "similarity_percent must be greater than 0 and at most 100"
            raise ValueError(msg)
        if self.maximum_edits is not None and self.maximum_edits < 0:
            msg = "maximum_edits cannot be negative"
            raise ValueError(msg)

    @classmethod
    def perfect_match(cls) -> FuzzinessOption:
        """Require source and proposed strings to be identical."""
        return cls(types=frozenset({FuzzinessType.PERFECT_MATCH}))

    @classmethod
    def whitespace_relaxation(cls) -> FuzzinessOption:
        """Permit only differences in whitespace runs."""
        return cls(types=frozenset({FuzzinessType.WHITESPACE_RELAXATION}))

    @classmethod
    def edit_distance(
        cls,
        *,
        similarity_percent: float | None = None,
        maximum_edits: int | None = None,
        whitespace_relaxation: bool = True,
    ) -> FuzzinessOption:
        """Permit a bounded edit distance, optionally after whitespace normalization."""
        types = {FuzzinessType.EDIT_DISTANCE}
        if whitespace_relaxation:
            types.add(FuzzinessType.WHITESPACE_RELAXATION)
        return cls(
            types=frozenset(types),
            similarity_percent=similarity_percent,
            maximum_edits=maximum_edits,
        )


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
    return _WHITESPACE.sub(" ", value).strip()
