"""Does relaxing the pin-cite whitespace recover the pin cites that fell into `extra`?

68 of 583 full case citations on this corpus carry a bare page number in
`extra` and no `pin_cite`. `550 U.S. 544` with `extra='570'` is a filing citing
page 570 of Twombly; the claim is on the page and the pipeline cannot see it.

The text after every one of them is doubled-spaced -- `'.  Legal  conclusions'`
-- which is the same defect that loses reference citations, in a third place.
`PIN_CITE_REGEX` spells its separators as a single optional literal space, so
`544,  570` does not parse as a pin cite and eyecite files the remainder as
`extra` instead.

This measures whether widening those spaces recovers them, and what else moves.
The widening is horizontal only: a doubled or tabbed separator matches, a
paragraph break does not.

    uv run python -m exploration.pinpoint.relax_pin_cites
"""

from __future__ import annotations

import argparse
import contextlib
import io
from collections import Counter
from pathlib import Path

import eyecite.helpers
import eyecite.regexes

from exploration.pinpoint.survey_extra import PARALLEL, PIN_SHAPED, body
from mellea_lrc.extraction import Relaxation, extract_from_plain_text

HORIZONTAL_OPTIONAL = r"[^\S\r\n]*"
HORIZONTAL_REQUIRED = r"[^\S\r\n]+"
CARRIES_PIN = ("FullCaseCitation", "ShortCaseCitation", "FullLawCitation", "FullJournalCitation")


def relaxed(pattern: str) -> str:
    """The same pattern with its literal spaces widened to horizontal whitespace."""
    return pattern.replace("\\ ?", HORIZONTAL_OPTIONAL).replace("\\ ", HORIZONTAL_REQUIRED)


@contextlib.contextmanager
def pin_cites_relaxed():
    """Swap in the widened patterns for the duration of the block.

    Both have to be swapped, and finding that out is the point of this file.
    `reference_pin_cite_re` reads `PIN_CITE_REGEX` when it is called, so
    patching that global reaches references. `POST_FULL_CITATION_REGEX` is an
    f-string that interpolates the same constant **at import time**, so the
    single-space version is already baked into it and patching the global does
    nothing for pin cites. `helpers.py` imports the composed pattern by value,
    so that name has to be patched too.
    """
    original_pin = eyecite.regexes.PIN_CITE_REGEX
    original_post = eyecite.helpers.POST_FULL_CITATION_REGEX
    eyecite.regexes.PIN_CITE_REGEX = relaxed(original_pin)
    eyecite.regexes.POST_FULL_CITATION_REGEX = relaxed(original_post)
    eyecite.helpers.POST_FULL_CITATION_REGEX = relaxed(original_post)
    try:
        yield
    finally:
        eyecite.regexes.PIN_CITE_REGEX = original_pin
        eyecite.regexes.POST_FULL_CITATION_REGEX = original_post
        eyecite.helpers.POST_FULL_CITATION_REGEX = original_post


def survey(documents: list[Path], level: Relaxation) -> tuple[Counter, dict]:
    """Count pin cites and record every citation's locator span and pin cite."""
    totals: Counter = Counter()
    detail: dict = {}

    for path in documents:
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=level)

        for citation in document.citations:
            kind = type(citation.citation).__name__
            totals[kind] += 1
            if kind not in CARRIES_PIN:
                continue
            pin = getattr(citation.citation, "pin_cite", None)
            extra = getattr(citation.citation, "extra", None)
            totals["with a pin cite"] += 1 if pin else 0
            written = str(extra).strip() if extra else ""
            if written and not PARALLEL.search(written) and PIN_SHAPED.match(written):
                totals["pin cite lost to extra"] += 1
            key = (path.stem[:14], citation.locator_span.start)
            detail[key] = (citation.locator_span.end, str(pin), written)
    return totals, detail


def main() -> int:
    """Compare extraction with and without the widened pin-cite regex."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path)
    parser.add_argument("--level", default="full", choices=[level.value for level in Relaxation])
    parser.add_argument("--show", type=int, default=12)
    args = parser.parse_args()

    directory = args.documents or Path("data/false-citation-bench-locator-only-v2.0/documents_txt")
    documents = sorted(directory.glob("*.txt"))
    level = Relaxation(args.level)
    print(f"{len(documents)} documents, relaxation={level.value}\n")

    before, before_detail = survey(documents, level)
    with pin_cites_relaxed():
        after, after_detail = survey(documents, level)

    print(f"{'':<28}{'before':>10}{'after':>10}{'change':>9}")
    for label in sorted(set(before) | set(after)):
        delta = after[label] - before[label]
        print(f"{label:<28}{before[label]:>10}{after[label]:>10}{(f'{delta:+d}' if delta else ''):>9}")

    spans_before = {k: v[0] for k, v in before_detail.items()}
    spans_after = {k: v[0] for k, v in after_detail.items()}
    print(f"\nlocator spans identical: {spans_before == spans_after}")
    print(f"  {len(spans_before)} before, {len(spans_after)} after")

    gained = [
        (k, before_detail[k], after_detail[k])
        for k in set(before_detail) & set(after_detail)
        if before_detail[k][1] != after_detail[k][1]
    ]
    print(f"\npin cites that changed: {len(gained)}, showing {args.show}")
    for (stem, start), was, now in sorted(gained)[: args.show]:
        print(f"  [{stem:<14}] at {start:>6}  pin {was[1]!r} -> {now[1]!r}   extra {was[2]!r} -> {now[2]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
