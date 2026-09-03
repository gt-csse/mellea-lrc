"""Which fields of a full case citation are present, and which look damaged.

Year and court are the next fields to be annotated, so this counts how often
each is there at all and flags the shapes that will need a rule.

    uv run python -m exploration.tree_bench.field_census
"""

from __future__ import annotations

import contextlib
import io
import re
from collections import Counter
from pathlib import Path

from exploration.tree_bench.census import DOCS, body
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction import Relaxation, extract_from_plain_text

YEAR = re.compile(r"\b(1[89]\d\d|20\d\d)\b")
AFTER = 46


def main() -> int:
    counts: Counter = Counter()
    year_shapes: Counter = Counter()
    missing_year_with_one_ahead = []
    for path in sorted(Path(DOCS).glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
        for item in document.citations:
            citation = item.citation
            if not isinstance(citation, FullCaseCitation):
                continue
            counts["full case citations"] += 1
            counts["year"] += bool(citation.year)
            counts["court"] += bool(citation.court)
            counts["parenthetical"] += bool(citation.parenthetical)
            counts["plaintiff"] += bool(citation.plaintiff)
            counts["defendant"] += bool(citation.defendant)
            counts["both parties"] += bool(citation.plaintiff and citation.defendant)
            tail = text[item.full_span.end : item.full_span.end + AFTER]
            if citation.year:
                year_shapes[str(citation.year)[:2] + "xx"] += 1
            else:
                ahead = YEAR.search(text[item.locator_span.end : item.locator_span.end + AFTER])
                if ahead:
                    missing_year_with_one_ahead.append(
                        (path.stem[:12], item.matched_text[:28], " ".join(tail.split())[:40])
                    )
    print(f"{'field':<24}{'present':>9}{'of':>7}")
    total = counts["full case citations"]
    for field in ("year", "court", "parenthetical", "plaintiff", "defendant", "both parties"):
        print(f"{field:<24}{counts[field]:>9}{total:>7}")
    print(f"\nyears by century: {dict(year_shapes)}")
    print(f"\n{len(missing_year_with_one_ahead)} citations carry no year while one stands just ahead")
    for stem, matched, tail in missing_year_with_one_ahead[:20]:
        print(f"  {stem:<14}{matched!r:<32}{tail!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
