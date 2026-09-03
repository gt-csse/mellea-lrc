r"""Is a case name recoverable by a widening rule, the way the others were?

Every patch this project has applied to eyecite had the same shape: the library
writes a literal character -- a space, a hyphen -- where PDF extraction leaves
something wider, so widening the pattern reads through the damage. Reporter
joins, pin cites, and the scan that finds the court and date were all this.

Case names are not. 108 of 583 case citations here carry an incomplete name: 80
with a defendant and no plaintiff, 28 with neither. Of those, 18 have a ` v. `
sitting in the text right before them, so a plaintiff is there and was missed.

Reading them, there are three causes and none of them is a separator.

**1. Damage inside the name.** `- ·Robinson v. N.C. Farm Bureau` loses its
plaintiff and the same sentence without the bullet keeps it; `Karim -Panahi v.
LAPD` loses its plaintiff and `Karim-Panahi` keeps it. Real, and fixable -- but
by repairing the text, which this project deliberately stopped doing, not by
widening a pattern.

**2. Punctuation inside a party name.** `Beery v. Hitachi Home Elecs. (Am.),
Inc.` comes back with `defendant='Am.), Inc.'`. The scan breaks on an opening
bracket. Clean text, no separator involved.

**3. The name is eaten before the scan runs.** `St. Amant v. Thompson, 390 U.S.
727` records `defendant='St. Amant'` -- the plaintiff in the defendant field,
Thompson gone. It is not the abbreviation and not the `citing` in front: dumping
the word tokens shows why::

    Thompson  ['Amant', ' ', 'v.', '390 U.S. 727', ',', ' ', '731', ' ']
    Howard    ['Amant', ' ', 'v.', ' ', 'Howard,', ' ', '390 U.S. 727', ',']

`Thompson,` is simply not in the token stream. It is a reporter in reporters-db,
and the tokenizer consumed it, so the backward scan meets `v.` immediately and
takes the plaintiff as the defendant.

## Why this is the case for a model

The first three patches worked because there was a pattern with a literal in it
and one substitution reached every instance. A case name is decided by a
stateful walk backwards over tokens, applying heuristics about capitalisation,
stop words and punctuation, over text a tokenizer has already had its way with.
There is no separator to widen, and each of the three causes would need its own
rule that a fourth case would break.

A reader given the sentence answers all three instantly, which is the argument
for the case-name re-extraction this project already has.

    uv run python -m exploration.court_and_date.probe_case_names
"""

from __future__ import annotations

import argparse
import contextlib
import io

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.reading.relaxation import tokenizer_for

CASES = [
    (
        "bullet and spacing, as extracted",
        "in: - ·Robinson v. N.C. Farm Bureau Ins. Co., 86 N.C. App. 44 (1987).",
    ),
    ("the same sentence, undamaged", "in Robinson v. N.C. Farm Bureau Ins. Co., 86 N.C. App. 44 (1987)."),
    ("a hyphen extraction has spaced", "See Karim -Panahi v. LAPD , 839 F.2d 621, 624 (9th Cir. 1988)."),
    ("the same name, undamaged", "See Karim-Panahi v. LAPD, 839 F.2d 621, 624 (9th Cir. 1988)."),
    ("a bracket inside the defendant", "in Beery v. Hitachi Home Elecs. (Am.), Inc., 157 F.R.D. 477 (1993)."),
    ("a defendant that is also a reporter", "Amant v. Thompson, 390 U.S. 727, 731 (1968)."),
    ("the same shape, another name", "Amant v. Jones, 390 U.S. 727, 731 (1968)."),
]


def main() -> int:
    """Print the parties parsed from each shape, and the tokens behind the last two."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for label, text in CASES:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
        citation = next(
            item.citation for item in document.citations if isinstance(item.citation, FullCaseCitation)
        )
        print(f"  {label:<38}plaintiff={citation.plaintiff!r:<18}defendant={citation.defendant!r}")

    print("\nthe tokens, for the name that disappears:")
    for name in ("Thompson", "Jones"):
        words, _ = tokenizer_for(Relaxation.FULL).tokenize(f"Amant v. {name}, 390 U.S. 727, 731 (1968).")
        print(f"  {name:<10}{[str(word) for word in list(words)[:8]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
