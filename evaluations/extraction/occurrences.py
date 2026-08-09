"""The public unit of extraction evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Occurrence:
    """One citation identifier and where it sits in a document.

    ``start`` and ``end`` are half-open offsets into the document body. Only
    the identifying fields are scored; ``matched_text`` and ``detail`` are
    carried into the evaluator's report untouched.
    """

    document: str
    start: int
    end: int
    matched_text: str | None = None
    detail: Mapping[str, object] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int, int]:
        """Return the document and span this occurrence sits at."""
        return (self.document, self.start, self.end)

    def to_row(self) -> dict[str, object]:
        """Render one JSONL row of the public run-artifact format."""
        row: dict[str, object] = {
            "document": self.document,
            "span": {"start": self.start, "end": self.end},
        }
        if self.matched_text is not None:
            row["matched_text"] = self.matched_text
        row.update(self.detail)
        return row


def deduplicate(occurrences: Iterable[Occurrence]) -> list[Occurrence]:
    """Keep the first occurrence of each span, preserving order."""
    seen: set[tuple[str, int, int]] = set()
    kept: list[Occurrence] = []
    for occurrence in occurrences:
        if occurrence.key in seen:
            continue
        seen.add(occurrence.key)
        kept.append(occurrence)
    return kept


def write_run_artifact(path: Path, occurrences: Iterable[Occurrence]) -> int:
    """Write the public JSONL run artifact and return how many rows it holds."""
    rows = [occurrence.to_row() for occurrence in occurrences]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return len(rows)
