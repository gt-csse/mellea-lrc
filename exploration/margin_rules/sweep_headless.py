"""Sweep for citations whose reporter and page are present but whose volume is not.

A Westlaw citation is `2016 WL 9137645`: the year is the volume. In a table of
authorities the table reader sometimes emits the columns out of order, so the
row reads `WL 9137645, at 3 (C.D. Cal. July 25, 2016)` with the year stranded
elsewhere. No tokenizer can read that -- eyecite needs the volume in front --
and the citation is invisible to any check that starts from what was extracted.

Site hunting finds them because the reporter is still spelled out, but it also
finds statutes and addresses, so this narrows to the one shape: a bare `WL`
followed by a docket-length number, with no year immediately before it.

The same sweep runs for a lettered reporter, where the volume is a plain number
rather than a year.

    uv run python -m exploration.margin_rules.sweep_headless
"""

from __future__ import annotations

import json
import re
from pathlib import Path

BENCH = Path("data/extraction-v2.0")

# `WL` with a number after it and no four-digit year immediately before it.
HEADLESS_WL = re.compile(r"(?<![\d\s]\d{3})(?<!\d)\s*\bWL\s+(\d{4,9})\b")
YEAR_BEFORE = re.compile(r"\b(19|20)\d{2}\s*$")


def body(path: Path) -> str:
    """The document text spans index into."""
    return path.read_text(encoding="utf-8")


def main() -> None:
    records = [
        json.loads(line) for line in (BENCH / "extraction.jsonl").read_text().splitlines() if line.strip()
    ]
    known: dict[str, list[tuple[int, int]]] = {}
    for record in records:
        known.setdefault(record["document"], []).append((record["span"]["start"], record["span"]["end"]))

    found = 0
    for path in sorted((BENCH / "documents_txt").glob("*.txt")):
        text = body(path)
        covered = known.get(path.name, [])
        for hit in re.finditer(r"\bWL\s+(\d{4,9})\b", text):
            if any(hit.start() < end and start < hit.end() for start, end in covered):
                continue
            before = text[max(0, hit.start() - 8) : hit.start()]
            if YEAR_BEFORE.search(before):
                continue
            found += 1
            start = max(0, hit.start() - 120)
            window = " ".join(text[start : hit.end() + 110].split())
            print(f"[{path.stem[:24]}]  WL {hit.group(1)}  at {hit.start()}")
            print(f"    {window!r}\n")
    print(f"WL citations with no year in front and no bench record: {found}")


if __name__ == "__main__":
    main()
