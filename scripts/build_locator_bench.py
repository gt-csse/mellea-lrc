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
        if len(hits) < len(ordered):
            problems.append(
                f"**dropped** {ordered[0]['document'][:40]}: {ordered[0]['matched_text']!r} "
                f"— the bench has {len(ordered)} occurrence(s), the text has {len(hits)}. "
                f"{len(ordered) - len(hits)} record(s) describe text that is not there."
            )
        elif len(hits) > len(ordered):
            # Not a loss. The text states this citation more often than the
            # published bench recorded it -- which is what happens once a margin
            # is removed and an interrupted occurrence reads cleanly, so the
            # same characters now match twice. Any surplus an INTERRUPTED entry
            # claims is added back below; the count at the end is what settles it.
            problems.append(
                f"surplus {ordered[0]['document'][:40]}: {ordered[0]['matched_text']!r} "
                f"— the text has {len(hits)} occurrence(s), the bench recorded {len(ordered)}. "
                f"{len(hits) - len(ordered)} left for an interrupted-locator entry to claim."
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


# Locators the published bench omitted because a page margin interrupts them.
# Its README calls these occurrences where "the filing states no complete
# identifier", and that reading is wrong: every character of the identifier is
# on the page, with a column of line numbers standing between the reporter and
# its page. Leaving them out takes the citation out of the denominator, so an
# extractor that never finds it still scores full recall -- which is exactly
# what happened before this list existed.
#
# Each is anchored at build time by the shape of the citation rather than by a
# stored offset, so the record survives the text being regenerated -- including
# being regenerated with the margin rule on, where the same citation is no
# longer interrupted at all and only a blank line separates the two halves.
INTERRUPTED = [
    {
        "document_prefix": "022",
        "volume": "214",
        "reporter": "F.3d",
        "page": "1058",
        "note": "the pleading-paper margin falls between the reporter and the page",
    },
]

# Locators whose reporter and page are on the page but whose volume is not in
# front of them. In a table of authorities the table reader can emit the
# columns out of order, and a Westlaw citation's volume is its year, so the row
# reads `WL 9137645, at 3 (C.D. Cal. July 25, 2016)` -- the year present, in the
# date parenthetical, and nowhere eyecite can use it.
#
# These are extraction misses, not absences. The filing states the citation;
# the converter delivered its parts in the wrong order.
# `exploration/notes/arm-disagreements-23aug.md` on the explorations branch
# reaches the same conclusion about the first of these and records that the gold
# keeps it. The published bench does not carry either.
#
# The span covers what is written -- the reporter and page -- because that is
# where an extractor that found this would report it. The volume is asserted
# from the row, so each entry names the year that has to be within `context`
# characters for the record to be built at all.
STRANDED_VOLUME = [
    {
        "document_prefix": "022",
        "volume": "2016",
        "reporter": "WL",
        "page": "9137645",
        "note": "table of authorities: the year sits in the date parenthetical, not before WL",
    },
    {
        "document_prefix": "025",
        "volume": "2023",
        "reporter": "WL",
        "page": "6200979",
        "note": "table of authorities: the year sits in the date parenthetical, not before WL",
    },
]
STRANDED_CONTEXT = 120


def gutter_pattern(volume: str, reporter: str, page: str) -> re.Pattern[str]:
    """Volume, reporter, any run of line numbers, then the page.

    Only whitespace and standalone integers may intervene. That is what a
    margin is; prose is not, so this cannot join two unrelated citations.

    The run may be empty, so the same entry anchors on text whose margin has
    been removed -- there the two halves are separated by the blank line the
    numbers used to sit in. Requiring at least one number would silently drop
    the record from a margin-adjusted bench, which is the failure this whole
    list exists to prevent.
    """
    reporter_chars = r"\s*".join(re.escape(c) for c in reporter if not c.isspace())
    return re.compile(rf"{re.escape(volume)}\s*{reporter_chars}(?:\s+\d{{1,3}})*\s+{re.escape(page)}")


def interrupted_records(text_dir: Path, already: list[dict]) -> tuple[list[dict], list[str]]:
    """Anchor each known margin-interrupted locator in the text.

    `already` is what the published records anchored to, and any hit overlapping
    one of those is that same occurrence rather than this one. The filing states
    this citation twice -- once in its table of authorities, where it reads
    cleanly and the published bench already carries it -- so without that check
    the loosened pattern would match the clean one too and the entry would be
    reported as ambiguous.
    """
    found: list[dict] = []
    problems: list[str] = []
    for entry in INTERRUPTED:
        matches = sorted(text_dir.glob(f"{entry['document_prefix']}*.txt"))
        if len(matches) != 1:
            problems.append(f"{entry['document_prefix']}*: matched {len(matches)} documents")
            continue
        document = matches[0]
        covered = [(r["span"]["start"], r["span"]["end"]) for r in already if r["document"] == document.name]
        hits = [
            hit
            for hit in gutter_pattern(entry["volume"], entry["reporter"], entry["page"]).finditer(
                body(document)
            )
            if not any(hit.start() < end and start < hit.end() for start, end in covered)
        ]
        if len(hits) != 1:
            problems.append(
                f"{document.name[:40]}: {entry['volume']} {entry['reporter']} {entry['page']} "
                f"interrupted by a margin -- expected 1, found {len(hits)}"
            )
            continue
        hit = hits[0]
        found.append(
            {
                "id": f"{entry['document_prefix']}:{hit.start()}-{hit.end()}",
                "document": document.name,
                "kind": "locator",
                "span": {"start": hit.start(), "end": hit.end()},
                "matched_text": hit.group(),
                "volume": entry["volume"],
                "reporter": entry["reporter"],
                "page": entry["page"],
                "source": "margin_interrupted",
                "note": entry["note"],
            }
        )
    return found, problems


def stranded_records(text_dir: Path, already: list[dict]) -> tuple[list[dict], list[str]]:
    """Anchor each locator whose volume is not in front of its reporter.

    The volume is only asserted when it is actually written nearby, so a record
    is built for a citation the filing states and refused for one it does not.
    """
    found: list[dict] = []
    problems: list[str] = []
    for entry in STRANDED_VOLUME:
        matches = sorted(text_dir.glob(f"{entry['document_prefix']}*.txt"))
        if len(matches) != 1:
            problems.append(f"{entry['document_prefix']}*: matched {len(matches)} documents")
            continue
        document = matches[0]
        text = body(document)
        covered = [(r["span"]["start"], r["span"]["end"]) for r in already if r["document"] == document.name]
        pattern = re.compile(rf"\b{re.escape(entry['reporter'])}\s+{re.escape(entry['page'])}\b")
        hits = [
            hit
            for hit in pattern.finditer(text)
            if not any(hit.start() < end and start < hit.end() for start, end in covered)
        ]
        if len(hits) != 1:
            problems.append(
                f"stranded-volume {document.name[:34]}: {entry['reporter']} {entry['page']} "
                f"— expected 1 unclaimed occurrence, found {len(hits)}"
            )
            continue
        hit = hits[0]
        window = text[max(0, hit.start() - STRANDED_CONTEXT) : hit.end() + STRANDED_CONTEXT]
        if entry["volume"] not in window:
            problems.append(
                f"stranded-volume {document.name[:34]}: {entry['reporter']} {entry['page']} "
                f"— volume {entry['volume']} not written within {STRANDED_CONTEXT} characters; "
                f"not treated as stated"
            )
            continue
        found.append(
            {
                "id": f"{entry['document_prefix']}:{hit.start()}-{hit.end()}",
                "document": document.name,
                "kind": "locator",
                "span": {"start": hit.start(), "end": hit.end()},
                "matched_text": hit.group(),
                "volume": entry["volume"],
                "reporter": entry["reporter"],
                "page": entry["page"],
                "source": "stranded_volume",
                "note": entry["note"],
            }
        )
    return found, problems


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
than excusing it.

That principle is why this bench carries one record the published one does not.
Document 022 states `214 F.3d 1058` twice: once in its table of authorities,
where it reads cleanly, and once in the argument, where the pleading-paper
margin falls between the reporter and the page. The published bench held only
the first, on the grounds that the second states no complete identifier -- but
every character of it is on the page, with a column of line numbers standing in
the middle. With it out of the denominator, an extractor that never finds it
scored 100% recall.

Two further cases are carried over from the published bench:

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

    extra, extra_problems = interrupted_records(args.text, anchored)
    anchored.extend(extra)
    problems.extend(extra_problems)

    stranded, stranded_problems = stranded_records(args.text, anchored)
    anchored.extend(stranded)
    problems.extend(stranded_problems)

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
        f"- margin-interrupted locators added: **{len(extra)}**",
        f"- stranded-volume locators added: **{len(stranded)}**",
        f"- locators anchored: **{len(anchored)}**",
        f"- unanchored: **{len(locators) + len(extra) + len(stranded) - len(anchored)}**",
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
