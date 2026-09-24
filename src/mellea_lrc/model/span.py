"""Span offsets in preprocessed document text."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Span:
    """Character offsets whose owning field identifies the referenced text."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            msg = f"Invalid span: start={self.start}, end={self.end}"
            raise ValueError(msg)

    def overlaps(self, other: Span) -> bool:
        """Whether two half-open spans share at least one character."""
        return (
            self.start < self.end
            and other.start < other.end
            and self.start < other.end
            and other.start < self.end
        )


def is_within(span: Span, regions: tuple[Span, ...]) -> bool:
    """Whether the whole span lies inside any region; abutting is not enough."""
    return any(region.start <= span.start and span.end <= region.end for region in regions)
