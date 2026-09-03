r"""What stopping at the co-location boundary would cost and buy.

The post-citation scan runs forward until a paragraph break, so a citation with
no parenthetical of its own takes the court and year of a later, unrelated
citation. Stopping at the next citation would fix that and break parallel
citations, whose single date sits after the *last* member.

So the boundary is the next citation **that is not co-located with this one**.
This re-runs eyecite's own post-citation pattern against the text up to that
boundary and reports what changes.

    uv run python -m exploration.court_and_date.simulate_boundary
"""

from __future__ import annotations

import argparse
import contextlib
import io
from collections import Counter
from pathlib import Path

import eyecite.regexes
import regex as re  # eyecite matches with this; it allows repeated group names
from eyecite.helpers import MAX_MATCH_CHARS, get_court_by_paren

from exploration.court_and_date.survey_missing import DOCS, body
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.reading.pin_cites import relax
from mellea_lrc.extraction.types import ExtractedCitation

# The same pattern eyecite matches after a citation, widened the way extraction
# already widens it, and anchored the way `match_on_tokens` anchors it.
BOUNDED = re.compile(rf"^(?:{relax(eyecite.regexes.POST_FULL_CITATION_REGEX)})", re.X)


def boundary(item: ExtractedCitation, citations: list[ExtractedCitation], length: int) -> int:
    """Where this citation's post-citation scan should stop.

    The next citation that is not co-located with it. A co-located one is
    another identifier for the same case, and the date belongs to the run as a
    whole, so reading across it is correct.
    """
    starts = [
        other.locator_span.start
        for other in citations
        if other is not item
        and other.locator_span.start >= item.locator_span.end
        and not (item.colocation_id and other.colocation_id == item.colocation_id)
    ]
    # eyecite never scans further than this either.
    return min([*starts, item.locator_span.end + MAX_MATCH_CHARS], default=length)


def main() -> int:
    """Compare the year and court eyecite records against a bounded re-read."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=DOCS)
    args = parser.parse_args()

    counts: Counter = Counter()
    lost: list[tuple[str, str, str]] = []
    dropped: list[tuple[str, str, str]] = []
    for path in sorted(args.documents.glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
        citations = list(document.citations)
        for item in citations:
            if not isinstance(item.citation, FullCaseCitation):
                continue
            counts["citations"] += 1
            stop = boundary(item, citations, len(text))
            window = text[item.locator_span.end : max(stop, item.locator_span.end)]
            found = BOUNDED.match(window)
            year = found.group("year") if found else None
            court_text = (found.group("court") or "").strip() if found else ""
            court = get_court_by_paren(court_text) if court_text else None

            was_year = item.citation.year
            counts["year now"] += bool(was_year)
            counts["year bounded"] += bool(year)
            if was_year and not year:
                counts["year lost"] += 1
                lost.append((path.stem[:10], item.matched_text[:20], str(was_year)))
            elif was_year and year and str(was_year) != str(year):
                counts["year changed"] += 1
                dropped.append((path.stem[:10], item.matched_text[:20], f"{was_year} -> {year}"))
            elif not was_year and year:
                counts["year gained"] += 1

            counts["court now"] += bool(item.citation.court)
            counts["court bounded"] += bool(court)

    for label, value in counts.items():
        print(f"  {label:<18}{value:>6}")
    print(f"\n{len(lost)} citations lose their year:")
    for row in lost:
        print(f"    {row[0]:<12}{row[1]:<22}was {row[2]}")
    print(f"\n{len(dropped)} citations change year:")
    for row in dropped:
        print(f"    {row[0]:<12}{row[1]:<22}{row[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
