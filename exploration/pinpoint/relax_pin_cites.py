r"""Does relaxing the pin-cite whitespace recover the pin cites that fell into `extra`?

68 of 583 full case citations on this corpus carried a bare page number in
`extra` and no `pin_cite`. `550 U.S. 544` with `extra='570'` is a filing citing
page 570 of Twombly: the claim is on the page and the pipeline cannot see it.

Two widenings take that to **one**, and both are the same defect -- eyecite
writes a literal character where PDF extraction leaves something wider.

**Spaces.** `PIN_CITE_REGEX` spells its separators as a single optional literal
space, so `544,  570` does not parse and the remainder is filed as `extra`.
Widening those to horizontal whitespace recovers 30.

**The range hyphen.** A page range is `\d+(?:-\d+)?`, with the hyphen against
the digits, and extraction spaces it: `998 -1003`, `337 - 38`, `189 - 90`.
Reading the 38 that survived the first widening, every one but a single
footnote cite was that shape. Allowing horizontal whitespace either side of the
hyphen, and an en dash beside the hyphen, recovers the rest.

    with a pin cite          387 -> 463   (+76)
    pin cite lost to extra    68 ->   1   (-67)
    every citation kind      unchanged except references
    locator spans            identical, 729 of 729

The one that remains is `928 F.3d 652, 657 n.1` -- a page followed by a
footnote. eyecite's pin-cite pattern allows a label *before* a page (`n. 5`) and
not after it, which is a different shape and not a whitespace problem.

Both patches must be applied in two places, and finding that out is worth
recording. `reference_pin_cite_re` reads `PIN_CITE_REGEX` when it is called, so
patching that global reaches references. `POST_FULL_CITATION_REGEX` is an
f-string interpolating the same constant **at import time**, so the strict
version is already baked in and patching the global does nothing for pin cites;
`helpers.py` imports the composed pattern by value, so that binding needs
patching too. Patching only the first looks exactly like a null result.

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


# A page range is written `\d+(?:-\d+)?`, with the hyphen against the digits.
# Extraction spaces it -- `998 -1003`, `337 - 38`, `189 - 90` -- and that one
# shape is every remaining lost pin cite on this corpus bar a single footnote
# cite.
RANGE_HYPHEN = r"[^\S\r\n]*[-–][^\S\r\n]*"


def relaxed(pattern: str) -> str:
    """The same pattern with its literal spaces and range hyphens widened.

    Two defects of the same kind: eyecite writes a literal single space where
    PDF extraction leaves several, and a bare hyphen where extraction leaves one
    with spaces around it.
    """
    widened = pattern.replace("\\ ?", HORIZONTAL_OPTIONAL).replace("\\ ", HORIZONTAL_REQUIRED)
    widened = widened.replace(r"(?:-\d+(?::\d+)?)?", rf"(?:{RANGE_HYPHEN}\d+(?::\d+)?)?")
    return widened.replace(r"(?:-\d+)?", rf"(?:{RANGE_HYPHEN}\d+)?")


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
