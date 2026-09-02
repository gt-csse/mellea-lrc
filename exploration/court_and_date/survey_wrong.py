r"""Where a year or court is not missing but wrong.

A missing year costs a check. A wrong one buys a confident verdict about the
wrong case, so this counts the second kind separately.

Two causes, both found by reading the code rather than guessed at.

**The post-citation scan does not stop at the next citation.** `add_post_citation`
calls `match_on_tokens` without `strings_only`, so it stops only at a paragraph
break, and `POST_FULL_CITATION_REGEX` spells its own `extra` group as `[^(;]*` --
unbounded until the next bracket or semicolon. A citation with no parenthetical
of its own therefore runs forward and takes the court and year belonging to a
later, unrelated citation. `Koulkina, 2009 WL 2103627, at *3.` two sentences
before `Spector v. Torenberg, 852 F. Supp. 201, 205 (S.D.N.Y. 1994)` comes back
carrying 1994.

Crossing the next citation is not always wrong. A parallel citation puts the
year after the *last* of the parallel forms, so `390 U.S. 727` reaching across
`88 S.Ct. 1323` for 1968 is correct. Co-location is what separates the two, and
it is already reported.

**Court names are matched by spelling.** `3rd Cir.` resolves and `3d Cir.` --
the form the Bluebook prescribes -- does not. `2d Cir.` resolves; `2nd Cir.`
resolves to the *Bankruptcy Appellate Panel*, which is a different court.

    uv run python -m exploration.court_and_date.survey_wrong
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

from eyecite.helpers import get_court_by_paren

from exploration.court_and_date.survey_missing import DOCS, body
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction import Relaxation, extract_from_plain_text

ORDINALS = ("1st Cir.", "2d Cir.", "2nd Cir.", "3d Cir.", "3rd Cir.", "2d Dept.", "1st Dept.")


def main() -> int:
    """Count citations whose year was taken from a different citation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=DOCS)
    args = parser.parse_args()

    parallel = suspect = total = 0
    rows = []
    for path in sorted(args.documents.glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
        citations = list(document.citations)
        for item in citations:
            if not isinstance(item.citation, FullCaseCitation):
                continue
            total += 1
            swallowed = [
                other
                for other in citations
                if other is not item
                and item.span.start < other.locator_span.start
                and other.locator_span.end <= item.span.end
                and isinstance(other.citation, FullCaseCitation)
            ]
            if not swallowed:
                continue
            # Same case in another reporter: reaching past it for the year is right.
            if all(o.colocation_id and o.colocation_id == item.colocation_id for o in swallowed):
                parallel += 1
                continue
            suspect += 1
            rows.append(
                (
                    path.stem[:10],
                    item.matched_text[:20],
                    item.citation.year,
                    swallowed[0].matched_text[:20],
                )
            )

    print(f"{total} full case citations")
    print(f"  {parallel} reach past a parallel citation for their year, which is correct")
    print(f"  {suspect} reach past an unrelated citation and take its year, which is not\n")
    for stem, matched, year, stolen in rows:
        print(f"  {stem:<12}{matched:<22}year={year!s:<6}taken from across {stolen!r}")

    print("\ncourt names, resolved by spelling:")
    for name in ORDINALS:
        print(f"  {name!r:<12} -> {get_court_by_paren(name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
