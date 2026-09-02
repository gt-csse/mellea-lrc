r"""Why a full case citation ends up with no year and no court.

Year is the strongest of the fields a filing states about a case: it is short,
it is unambiguous, and it is what tells a lookup which of several cases with the
same party names is meant. eyecite finds one for 518 of 583 citations here.

This prints the text right after every citation that has none, so the causes can
be read rather than guessed. What matters is which of them are whitespace and
which are structural, because only the first kind is fixable the way the
reporter joins and pin cites were.

    uv run python -m exploration.court_and_date.survey_missing
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
from collections import Counter
from pathlib import Path

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction import Relaxation, extract_from_plain_text

BODY_MARKER = "--- Plain text ---\n"
DOCS = Path("data/false-citation-bench-locator-only-v2.0/documents_txt")
AFTER = 54
YEAR = re.compile(r"\b(1[7-9]\d\d|20\d\d)\b")


def body(path: Path) -> str:
    """The document text the bench spans index into."""
    text = path.read_text(encoding="utf-8")
    _, marker, rest = text.partition(BODY_MARKER)
    return rest if marker else text


def cause(tail: str) -> str:
    """A first guess at why the parse failed, read from the characters."""
    found = YEAR.search(tail)
    if not found:
        return "no year within reach"
    before_year = tail[: found.start()]
    if "(" not in before_year and "[" not in before_year:
        return "no opening bracket"
    closing = tail[found.end() : found.end() + 4]
    if re.match(r"[^\S\r\n]+[\)\]]", closing):
        return "space before the closing bracket"
    if re.search(r"[,;][^\S\r\n]*$", before_year):
        return "punctuation between court and year"
    if re.search(r"[A-Za-z.][^\S\r\n]*$", before_year) and not before_year.endswith(" "):
        return "no space between court and year"
    return "other"


def main() -> int:
    """Print every full case citation carrying no year, with the text after it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=DOCS)
    parser.add_argument("--show", type=int, default=70)
    args = parser.parse_args()

    causes: Counter = Counter()
    rows = []
    totals = Counter()
    for path in sorted(args.documents.glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
        # A year belonging to the *next* citation is not this one's, so the
        # text examined stops where the next citation begins -- the same bound
        # the tree bench uses when reading a pin cite.
        starts = sorted(c.locator_span.start for c in document.citations)
        for item in document.citations:
            citation = item.citation
            if not isinstance(citation, FullCaseCitation):
                continue
            totals["citations"] += 1
            totals["year"] += bool(citation.year)
            totals["court"] += bool(citation.court)
            if citation.year:
                continue
            end = item.locator_span.end
            boundary = min((s for s in starts if s >= end), default=len(text))
            tail = text[end : min(end + AFTER, boundary)]
            reason = cause(tail)
            causes[reason] += 1
            rows.append((reason, path.stem[:10], item.matched_text[:22], " ".join(tail.split())))

    print(
        f"{totals['citations']} full case citations: {totals['year']} with a year, {totals['court']} with a court\n"
    )
    print(f"{totals['citations'] - totals['year']} with no year, by apparent cause:")
    for reason, count in causes.most_common():
        print(f"  {count:>4}  {reason}")
    print()
    for reason, stem, matched, tail in sorted(rows)[: args.show]:
        print(f"  [{reason[:34]:<34}] {stem:<12}{matched:<24}{tail[:52]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
