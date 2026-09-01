"""Build false-citation-bench-locator-only-v1.0 from the published bench and v1.1.

The published bench mixes two things this project checks differently. Of its 594
occurrences, 11 are docket numbers -- `No. 1:19-CV-362` and the like -- which
eyecite does not attempt at all and which put a floor of 11 false negatives
under every arm, capping recall at 98.1% for reasons that have nothing to do
with how well citations are read. The other 583 are locators: a volume, a
reporter and a page, which is what verification actually resolves.

This writes a bench of the locators alone, anchored to the v1.1 text.

**Ground truth is inclusive.** A locator the filing states is ground truth even
when no tokenizer reaches it, and an extractor that misses it has missed it.
Two such are already carried by the published bench and are kept here:

    759\\n\\nF.2d 1032    a citation split across a blank line
    455 US. 363         a reporter written without the period after US

The second is worth noting for what it says about the text rather than the
tokenizer: Docling 2.115.0 renders that same page as `455 U.S. 363`, so on v1.1
it is no longer hard. The record stays because the citation is the same
citation; what changed is the converter under it.

**Anchoring is by content, not by offset.** v1 and v1.1 came from different
Docling versions, so every offset moved, and 17 locators are not even findable
verbatim -- the newer converter inserts the space that `2016 WL1448829` was
missing, and writes `NY Slip Op` where the older one wrote `NYSlip Op`. So each
record is re-found by its characters with any whitespace allowed between them,
which reaches all three shapes, and occurrences of one locator are matched to
candidates in document order.

    uv run python scripts/build_locator_bench.py

Anything that could not be anchored is written to `build-report.md` rather than
guessed at, because a bench with a wrong span is worse than one with a gap.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

BODY_MARKER = "--- Plain text ---\n"


def body(path: Path) -> str:
    """The document body: the text after the provenance header."""
    text = path.read_text(encoding="utf-8")
    _, marker, rest = text.partition(BODY_MARKER)
    return rest if marker else text


def loose(literal: str) -> re.Pattern[str]:
    """The same characters, with any whitespace allowed between each of them.

    Whitespace inside the literal is dropped rather than matched, so this reads
    `2016 WL1448829` and `2016 WL 1448829` and `759\\n\\nF.2d 1032` as one
    pattern. Only whitespace is skipped, so it cannot run past the citation
    into neighbouring text.
    """
    return re.compile(r"\s*".join(re.escape(c) for c in literal if not c.isspace()))


def loose_locator(record: dict) -> re.Pattern[str]:
    """The record's volume, reporter and page, with the punctuation optional too.

    The fallback for when the literal text cannot be found because the
    converter changed a character rather than a space. `455 US. 363` is written
    `455 U.S. 363` by Docling 2.115.0 -- a period appears inside the reporter,
    which no amount of whitespace tolerance reaches. Every period in the
    reporter is therefore optional here.

    This is deliberately not the primary path. It identifies a citation by what
    it says rather than by how it was written, which is right for re-finding a
    known record and would be far too loose for discovering a new one.
    """
    parts = [re.escape(record["volume"])]
    for character in record["reporter"]:
        if character.isspace():
            continue
        parts.append(r"\.?" if character == "." else re.escape(character))
    parts.append(re.escape(record["page"]))
    return re.compile(r"\s*".join(parts))


def anchor(records: list[dict], text: str) -> tuple[list[dict], list[str]]:
    """Re-find each record in `text`, matching repeats in document order."""
    anchored: list[dict] = []
    problems: list[str] = []

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped["".join(record["matched_text"].split())].append(record)

    for key, group in grouped.items():
        ordered = sorted(group, key=lambda r: r["span"]["start"])
        hits = list(loose(key).finditer(text))
        note = None
        if len(hits) < len(ordered):
            relaxed = list(loose_locator(ordered[0]).finditer(text))
            if len(relaxed) >= len(ordered):
                hits = relaxed
                note = "anchored on the locator rather than the written text"
        if len(hits) != len(ordered):
            problems.append(
                f"{ordered[0]['document'][:40]}: {ordered[0]['matched_text']!r} "
                f"expected {len(ordered)} occurrence(s), found {len(hits)} "
                f"-- {min(len(ordered), len(hits))} kept, {len(ordered) - len(hits)} dropped"
            )
        for record, hit in zip(ordered, hits, strict=False):
            anchored.append(
                {
                    **record,
                    "span": {"start": hit.start(), "end": hit.end()},
                    "matched_text": hit.group(),
                    "id": f"{record['document'][:3]}:{hit.start()}-{hit.end()}",
                    **({"anchor_note": note} if note else {}),
                }
            )
    return anchored, problems


README = """# false-citation-bench, locator only (v1.0)

The same text as `../false-citation-bench-v1.1/`, with a ground truth that
holds **locators only** -- a volume, a reporter and a page.

## What changed from the published bench

The published `derived/extraction.jsonl` carries 594 occurrences: 583 locators
and 11 docket numbers. The dockets are dropped here. eyecite does not attempt
them, so they placed a floor of 11 false negatives under every extraction arm
and capped recall at 98.1% for a reason unrelated to how well citations are
read. Nothing about them is a judgement on extraction, and mixing them in made
every recall figure need a footnote.

Every span is re-anchored to the v1.1 text. v1 and v1.1 came from different
Docling versions, so no offset from the published bench transfers.

## The ground truth is inclusive

A locator the filing states is ground truth even when no tokenizer reaches it.
An extractor that does not find it has missed it, and the bench says so rather
than excusing it. Two such cases are carried over from the published bench:

- `759\\n\\nF.2d 1032` -- split across a blank line
- `455 US. 363` -- a reporter written without the period after `US`

The second reads as `455 U.S. 363` in this text: Docling 2.115.0 renders that
page differently from the version that produced v1. The record is the same
citation either way.

## What is not here

Docket numbers, statutes, regulations and journal citations. This bench answers
one question -- did extraction find the case locators the document states --
and anything else belongs in a bench of its own.

Produced by `scripts/build_locator_bench.py` on branch `extraction/relaxation`.
See `build-report.md` for anything that could not be anchored. Nothing here is
published.
"""


def main() -> int:
    """Filter, re-anchor and write the locator-only bench."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench", type=Path, default=Path("data/false-citation-bench/derived/extraction.jsonl")
    )
    parser.add_argument("--text", type=Path, default=Path("data/false-citation-bench-v1.1/documents_txt"))
    parser.add_argument("--out", type=Path, default=Path("data/false-citation-bench-locator-only-v1.0"))
    args = parser.parse_args()

    records = [json.loads(line) for line in args.bench.read_text().splitlines() if line.strip()]
    locators = [r for r in records if r["kind"] == "locator"]
    dropped = len(records) - len(locators)

    by_document: dict[str, list[dict]] = defaultdict(list)
    for record in locators:
        by_document[record["document"]].append(record)

    anchored: list[dict] = []
    problems: list[str] = []
    for document, group in sorted(by_document.items()):
        source = args.text / document
        if not source.exists():
            problems.append(f"{document}: absent from {args.text}")
            continue
        found, issues = anchor(group, body(source))
        anchored.extend(found)
        problems.extend(issues)

    anchored.sort(key=lambda r: (r["document"], r["span"]["start"]))

    text_out = args.out / "documents_txt"
    text_out.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.text.glob("*.txt")):
        shutil.copy2(path, text_out / path.name)

    (args.out / "extraction.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in anchored),
        encoding="utf-8",
    )
    (args.out / "README.md").write_text(README, encoding="utf-8")

    report = [
        "# Building the locator-only bench",
        "",
        f"- published records: **{len(records)}**",
        f"- docket records dropped: **{dropped}**",
        f"- locators to anchor: **{len(locators)}**",
        f"- locators anchored: **{len(anchored)}**",
        f"- unanchored: **{len(locators) - len(anchored)}**",
        "",
    ]
    if problems:
        report += ["## Needs reading", "", *(f"- {problem}" for problem in problems), ""]
    else:
        report += ["Every locator anchored cleanly.", ""]
    (args.out / "build-report.md").write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
