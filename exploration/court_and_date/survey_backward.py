r"""Does the backward scan cross a citation the way the forward one did?

The forward scan took the court and date of the next citation because
``add_post_citation`` never stopped at one. The backward scan reads the case
name, and the question is whether it has the same defect.

Two of eyecite's three backward paths are bounded. ``add_pre_citation`` passes
``strings_only=True``, so it stops at any non-string token -- a preceding
citation included -- and its pattern is one capitalised word besides.

``find_case_name`` is not. Scanning back from the citation it reaches this::

    # Handle citation tokens - just adjust the title boundary
    if isinstance(word, CitationToken):
        state["title_starting_index"] = index - 1
        continue

A preceding citation moves the boundary and the scan carries on, up to
``BACKWARD_SEEK`` (28) tokens. So it can read a case name from the far side of
another citation.

This counts how often a citation's span reaches back across one, and prints the
party names it came away with, because unlike a year a case name has no
arithmetic check -- whether it is the right name has to be read.

## Crossing is deliberate, and necessary

A parallel citation carries its case name once, in front of the *first* of the
parallel forms, so every later member has to read back across it. All 16
non-first members of a parallel group on this corpus get their name that way.
Stopping at a citation would cost all 16 to save the 2, so the behaviour stays.

(`BACKWARD_SEEK` is a different thing. Its own comment says 28 is the median
case name length in the CourtListener database -- a budget for how long a name
may be, not a statement about crossing.)

## What the same measurement found instead

Case names are unreliable well beyond the two crossings::

    both parties            475 of 583
    defendant only           80
    neither                  28
    plaintiff is `""`        11   -- an empty string rather than None

And they can be silently wrong. `citing St. Amant v. Thompson, 390 U.S. 727` is
recorded with `plaintiff=""` and `defendant="St. Amant"`: the plaintiff is in
the defendant field and Thompson is gone. `Garrison v. Louisiana` in the very
next sentence parses correctly, so what breaks it is the period inside `St.
Amant` rather than anything about the sentence.

That is roughly 18% incomplete before counting the swaps, which is the evidence
for annotating a pin cite before a case name: a pin cite is a fact about the
text, and a case name is already wrong here often enough to need checking on its
own terms.

    uv run python -m exploration.court_and_date.survey_backward
"""

from __future__ import annotations

import argparse
import contextlib
import io
from collections import Counter
from pathlib import Path

from exploration.court_and_date.survey_missing import DOCS, body
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction import Relaxation, extract_from_plain_text


def main() -> int:
    """Print every citation whose span reaches back over another citation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=DOCS)
    parser.add_argument("--show", type=int, default=40)
    args = parser.parse_args()

    counts: Counter = Counter()
    rows = []
    for path in sorted(args.documents.glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
        citations = list(document.citations)
        for item in citations:
            citation = item.citation
            if not isinstance(citation, FullCaseCitation):
                continue
            counts["citations"] += 1
            counts["names a party"] += bool(citation.plaintiff or citation.defendant)
            crossed = [
                other
                for other in citations
                if other is not item
                # Entirely before this citation's locator, and inside its span.
                and other.locator_span.end <= item.locator_span.start
                and other.locator_span.start >= item.span.start
                and not (item.colocation_id and other.colocation_id == item.colocation_id)
            ]
            if not crossed:
                continue
            counts["span reaches back over a citation"] += 1
            rows.append(
                (
                    path.stem[:10],
                    item.matched_text[:20],
                    f"{citation.plaintiff} v. {citation.defendant}",
                    " ".join(text[item.span.start : item.locator_span.start].split())[-72:],
                )
            )

    for label, value in counts.items():
        print(f"  {label:<38}{value:>5}")
    print()
    for stem, matched, parties, before in rows[: args.show]:
        print(f"  [{stem:<10}] {matched:<22}{parties[:44]!r}")
        print(f"               reaching back over: {before!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
