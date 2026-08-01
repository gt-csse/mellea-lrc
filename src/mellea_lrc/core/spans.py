"""Span offsets in preprocessed document text."""

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
