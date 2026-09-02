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
