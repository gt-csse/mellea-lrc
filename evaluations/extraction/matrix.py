r"""What each mechanism is worth, measured one at a time on both corpora.

Each row turns off exactly one thing the composite does and reports the same
columns, so a mechanism's value is the difference between its row and the one
above it. Two properties make this a reliability matrix rather than a volume
one:

*   Half the columns count **defects**, not findings. A configuration that reads
    more citations while carrying more wrong years is worse, and the table says
    so instead of hiding it behind a recall number.
*   Recall is scored against ground truth where it exists. The 586 locators of
    `false-citation-bench-locator-only-v2.0` are annotated and inclusive: a
    locator the filing states counts whether or not any tokenizer reaches it.
    The 77 mined filings have no ground truth, so their columns are counts and
    defect counts only, never recall.

Site hunting is deliberately absent. It proposes 185 candidates on the mined
corpus to reach at most 2 real citations, and its cost is a reviewer's rather
than a measurement's.

    uv run python -m evaluations.extraction.matrix
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from mellea_lrc.core.citations import CitationKind, FullCaseCitation
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction import eyecite_extractor as extractor_module
from mellea_lrc.extraction import stages as stages_module
from mellea_lrc.extraction.reading import pin_cites as pin_cites_module
from mellea_lrc.extraction.reading import post_citation as post_citation_module
from mellea_lrc.extraction.types import ExtractedDocument

BENCH = Path("data/extraction-v2.0/documents_txt")
BENCH_TRUTH = Path("data/extraction-v2.0/locators.jsonl")
MINED = Path.home() / "CodingProjects/mellea-lrc/local/mined-corpus"

# A pin cite eyecite could not place: digits, star pages, paragraph marks in
# `extra`. Not a section, which names a statute, and not a parallel citation.
PIN_SHAPED = re.compile(r"^[*¶]?\s*\d[\d\s,\-–*¶n.]*$")
PARALLEL = re.compile(r"\d+\s+[A-Z][A-Za-z.'’ ]*\d*\s+\d+")
# A court written in a parenthetical, which the citation should have recorded.
COURT_IN_PAREN = re.compile(r"\(([^)]{2,40}?)[^\S\r\n]+(?:1[7-9]\d\d|20\d\d)[^\S\r\n]*\)")


def body(path: Path) -> str:
    """The document text spans index into."""
    return path.read_text(encoding="utf-8")


@dataclass(frozen=True, slots=True)
class Arm:
    """One configuration: a relaxation level and a mechanism switched off."""

    name: str
    relaxation: Relaxation = Relaxation.FULL
    disable: str = ""
    """Which mechanism this row removes, relative to the full composite."""


ARMS: tuple[Arm, ...] = (
    Arm("eyecite as published", relaxation=Relaxation.NONE, disable="everything"),
    Arm("+ reporter joins (bounded)", relaxation=Relaxation.BOUNDED, disable="everything"),
    Arm("+ reporter joins (full)", relaxation=Relaxation.FULL, disable="everything"),
    Arm("+ pin cites", disable="post_citation,courts,dockets"),
    Arm("+ court and date boundary", disable="courts,dockets"),
    Arm("+ court resolution", disable="dockets"),
    Arm("+ docket locators  = the composite", disable=""),
)


@contextlib.contextmanager
def _configured(arm: Arm) -> Iterator[None]:
    """Run the real pipeline with this arm's mechanisms removed.

    Removal is by patching rather than by reimplementing the pipeline, so every
    row is the shipped code path with one piece taken out.

    **Each name is patched where it is looked up, not where it is defined.**
    This project's modules import functions by name, so replacing
    `post_citation.reread_post_citation` leaves `stages` holding the original
    and silently measures nothing -- which this matrix did on its first run, and
    which is the same trap `POST_FULL_CITATION_REGEX` set earlier.
    """
    off = {name for name in arm.disable.split(",") if name}
    everything = "everything" in off
    patches: list[tuple[object, str, object]] = []

    if everything or "pin_cites" in off:
        patches.append((pin_cites_module, "relax", lambda pattern: pattern))
    if everything or "post_citation" in off:
        # Looked up in `stages`, which imported it by name.
        patches.append((stages_module, "reread_post_citation", lambda _text, cites: tuple(cites)))
    if everything or "courts" in off:
        # Looked up inside `post_citation`, which is where it is called.
        from eyecite.helpers import get_court_by_paren

        patches.append((post_citation_module, "resolve_court", get_court_by_paren))
    if everything or "dockets" in off:
        # Looked up in the extractor, which imported it by name.
        patches.append((extractor_module, "with_dockets", lambda tokenizer: tokenizer))

    saved = [(owner, attribute, getattr(owner, attribute)) for owner, attribute, _ in patches]
    try:
        for owner, attribute, replacement in patches:
            setattr(owner, attribute, replacement)
        yield
    finally:
        for owner, attribute, original in saved:
            setattr(owner, attribute, original)


def _court_written_but_unrecorded(text: str, document: ExtractedDocument) -> int:
    """Citations whose parenthetical names a court that was not recorded."""
    missed = 0
    for item in document.citations:
        citation = item.citation
        if not isinstance(citation, FullCaseCitation) or citation.court:
            continue
        tail = text[item.locator_span.end : item.locator_span.end + 70]
        if COURT_IN_PAREN.search(tail):
            missed += 1
    return missed


def _measure(text: str, document: ExtractedDocument) -> dict[str, int]:
    """Findings and defects for one document under one arm."""
    counts = dict.fromkeys(
        (
            "case citations",
            "dockets",
            "with a pin cite",
            "pin cite lost",
            "with a date",
            "date taken from another case",
            "with a court",
            "court written, not recorded",
            "reporter with no edition",
        ),
        0,
    )
    citations = list(document.citations)
    for item in citations:
        citation = item.citation
        if citation.kind is CitationKind.DOCKET:
            counts["dockets"] += 1
        if not isinstance(citation, FullCaseCitation):
            continue
        counts["case citations"] += 1
        counts["with a pin cite"] += bool(citation.pin_cite)
        counts["with a date"] += bool(citation.date)
        counts["with a court"] += bool(citation.court)
        # No edition means no canonical name, cite type or scotus flag, and it
        # is silent: eyecite leaves `edition_guess` unset when the year filter
        # empties a list of candidates rather than falling back to it.
        counts["reporter with no edition"] += bool(citation.reporter and citation.reporter.short_name is None)

        extra = (citation.extra or "").strip()
        if extra and not PARALLEL.search(extra) and PIN_SHAPED.match(extra):
            counts["pin cite lost"] += 1

        # A span reaching over an unrelated citation took that citation's
        # court and date. Co-located neighbours are excluded: reading across
        # one is what a parallel citation requires.
        if citation.date and any(
            other is not item
            and item.locator_span.end <= other.locator_span.start
            and other.locator_span.end <= item.span.end
            and not (item.colocation_id and other.colocation_id == item.colocation_id)
            for other in citations
        ):
            counts["date taken from another case"] += 1

    counts["court written, not recorded"] = _court_written_but_unrecorded(text, document)
    return counts


def _stated_locators() -> dict[str, set[tuple[int, int]]]:
    """The annotated ground truth: every locator the bench filings state."""
    stated: dict[str, set[tuple[int, int]]] = {}
    for line in BENCH_TRUTH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            stated.setdefault(record["document"], set()).add((record["span"]["start"], record["span"]["end"]))
    return stated


def run(directory: Path, arm: Arm, *, truth: dict[str, set[tuple[int, int]]] | None) -> dict[str, int]:
    """Measure one arm over one corpus."""
    totals: dict[str, int] = {}
    found: dict[str, set[tuple[int, int]]] = {}
    with _configured(arm):
        for path in sorted(directory.glob("*.txt")):
            text = body(path)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                document = extract_from_plain_text(text, relaxation=arm.relaxation)
            for key, value in _measure(text, document).items():
                totals[key] = totals.get(key, 0) + value
            if truth is not None:
                found[path.name] = {
                    (item.locator_span.start, item.locator_span.end)
                    for item in document.citations
                    if isinstance(item.citation, FullCaseCitation)
                }
    if truth is not None:
        stated = {(d, s, e) for d, spans in truth.items() for s, e in spans}
        got = {(d, s, e) for d, spans in found.items() for s, e in spans}
        totals["locators stated"] = len(stated)
        totals["locators found"] = len(stated & got)
        totals["spurious"] = len(got - stated)
    return totals


def main() -> int:
    """Print the matrix for both corpora."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=BENCH)
    parser.add_argument("--mined", type=Path, default=MINED)
    args = parser.parse_args()

    truth = _stated_locators()
    for label, directory, ground in (
        ("false-citation-bench (586 locators annotated)", args.bench, truth),
        ("mined corpus, unseen (no ground truth)", args.mined, None),
    ):
        if not directory.exists():
            print(f"\n{label}: not found at {directory}")
            continue
        print(f"\n## {label}\n")
        columns = (["locators found", "spurious"] if ground else ["case citations"]) + [
            "dockets",
            "with a pin cite",
            "pin cite lost",
            "with a date",
            "date taken from another case",
            "with a court",
            "court written, not recorded",
            "reporter with no edition",
        ]
        header = f"{'arm':<38}" + "".join(f"{c[:13]:>15}" for c in columns)
        print(header)
        print("-" * len(header))
        for arm in ARMS:
            totals = run(directory, arm, truth=ground)
            print(f"{arm.name:<38}" + "".join(f"{totals.get(c, 0):>15}" for c in columns))
    print("\nfindings on the left of each pair, defects on the right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
