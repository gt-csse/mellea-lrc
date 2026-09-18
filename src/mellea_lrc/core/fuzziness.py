"""Project-wide policies and helpers for forgiving text layout variation.

The policy is shared by two different operations. :func:`fuzzy_literal`
compiles a policy into a regex fragment for discovery, while source-grounding
uses the same policy to compare a model's proposed string with known evidence.
An edit-distance policy is meaningful only for the latter: an open-ended regex
cannot safely discover every string within an edit-distance threshold.
"""

from __future__ import annotations

import re
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
    compose when comparing known candidate strings; only whitespace relaxation
    can be compiled into a safe open-ended literal regex.
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


def fuzzy_literal(
    value: str,
    fuzziness: FuzzinessOption,
    *,
    newline: bool = False,
) -> str:
    """Compile a literal string under a discovery-safe fuzziness policy.

    Whitespace relaxation permits arbitrary whitespace around literal
    punctuation as well as at written whitespace joins. Non-whitespace
    characters remain literal. Edit distance needs a fixed candidate value and
    is therefore intentionally unavailable for regex discovery.
    """
    if FuzzinessType.EDIT_DISTANCE in fuzziness.types:
        msg = "fuzzy_literal cannot compile an EDIT_DISTANCE policy"
        raise ValueError(msg)
    if fuzziness.types == frozenset({FuzzinessType.PERFECT_MATCH}):
        return re.escape(value)
    if FuzzinessType.WHITESPACE_RELAXATION not in fuzziness.types:
        msg = f"fuzzy_literal cannot compile {sorted(kind.value for kind in fuzziness.types)!r}"
        raise ValueError(msg)

    separator = r"\s*" if newline else r"[^\S\r\n]*"
    tokens = tuple(re.finditer(r"\w+|[^\w\s]+", value))
    if not tokens:
        return separator
    pattern = ""
    for index, token in enumerate(tokens):
        pattern += re.escape(token.group())
        if index == len(tokens) - 1:
            if value[token.end() :]:
                pattern += separator
            continue
        following = tokens[index + 1]
        if (
            value[token.end() : following.start()]
            or not token.group().isalnum()
            or not following.group().isalnum()
        ):
            pattern += separator
    return pattern
