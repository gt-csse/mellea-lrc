"""Explicit policies for grounding proposed text to known source evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FuzzinessType(str, Enum):
    """One permitted text-comparison operation."""

    PERFECT_MATCH = "perfect_match"
    WHITESPACE_RELAXATION = "whitespace_relaxation"
    EDIT_DISTANCE = "edit_distance"


@dataclass(frozen=True, slots=True)
class FuzzinessOption:
    """An explicit, composable policy for comparing or finding text.

    ``PERFECT_MATCH`` is exclusive. Whitespace relaxation and edit distance
    compose when comparing known candidate strings. A regex cannot safely
    discover every string within an edit-distance threshold.
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
        """Permit arbitrary whitespace insertion, removal, or run variation only."""
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
