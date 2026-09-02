"""How often do full spans coincide on the real corpus, and what is in those groups?

Co-location is proposed as the extraction-layer signal for a parallel citation:
citations whose full spans coincide are a candidate group, and validation
decides whether they name one case. A candidate signal is only useful if the
groups it forms are mostly real, so this counts them.

One test can be applied here without any lookup, and it is the only one that
can: **two citations sharing a reporter are two cases**, because a case has one
first page in one reporter. Any group containing a repeated reporter is a group
the signal got wrong, and it can be counted without asking CourtListener
anything.

Groups of distinct reporters are candidates, not answers. Whether they name one
case is validation's question.

    uv run python -m exploration.pinpoint.colocation_census
"""

from __future__ import annotations

import argparse
import contextlib
import io
from collections import Counter
from pathlib import Path

from exploration.pinpoint.survey_extra import body
from mellea_lrc.extraction import Relaxation, extract_from_plain_text

FULL_KINDS = ("FullCaseCitation", "FullLawCitation", "FullJournalCitation")


def groups(citations: list) -> list[list]:
    """Citations sharing a full span, as groups of two or more."""
    by_span: dict[tuple[int, int], list] = {}
    for citation in citations:
        by_span.setdefault((citation.span.start, citation.span.end), []).append(citation)
    return [group for group in by_span.values() if len(group) > 1]


def main() -> int:
    """Count co-location groups and how many are provably not one case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path)
    parser.add_argument("--show", type=int, default=20)
    args = parser.parse_args()

    documents = args.documents or Path("data/false-citation-bench-locator-only-v2.0/documents_txt")
    totals: Counter = Counter()
    wrong: list[tuple[str, list[str]]] = []
    candidates: list[tuple[str, list[str]]] = []

    for path in sorted(documents.glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)

        full = [c for c in document.citations if type(c.citation).__name__ in FULL_KINDS]
        totals["full citations"] += len(full)

        for group in groups(full):
            totals["co-location groups"] += 1
            totals["citations in a group"] += len(group)
            written = [" ".join(c.matched_text.split()) for c in group]
            reporters = [str(getattr(c.citation, "reporter", "")) for c in group]
            if len(set(reporters)) < len(reporters):
                totals["groups with a repeated reporter -- not one case"] += 1
                wrong.append((path.stem[:18], written))
            else:
                totals["groups of distinct reporters -- candidates"] += 1
                candidates.append((path.stem[:18], written))

    print("| | count |")
    print("|---|---:|")
    for label in (
        "full citations",
        "co-location groups",
        "citations in a group",
        "groups of distinct reporters -- candidates",
        "groups with a repeated reporter -- not one case",
    ):
        print(f"| {label} | {totals[label]} |")

    print(f"\n--- groups the signal got wrong: {len(wrong)} ---")
    for stem, written in wrong[: args.show]:
        print(f"  [{stem:<18}] {written}")

    print(f"\n--- candidate groups: {len(candidates)}, showing {args.show} ---")
    for stem, written in candidates[: args.show]:
        print(f"  [{stem:<18}] {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
