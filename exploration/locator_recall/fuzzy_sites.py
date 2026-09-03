"""Find every number-string-number in the corpus, however badly spelled.

Site hunting asks where a reporter the gazetteer *knows* appears with digits on
both sides. It is recall-oriented but it still starts from an exact string, so
it cannot see a reporter the converter has misspelled -- and a misspelled
reporter is precisely the case no tokenizer reaches, which is the case worth
finding.

This asks the weaker question, and asks it of the whole corpus:

    a number, then some letters, then a number

Everything between the parts may be any run of non-alphanumeric characters, so
`214 F.3d 1058`, `2016WL9137645`, `455 US. 363` and `937 | S.W.2d | 796` are
all one shape. The letters are then reduced to letters alone -- `F. Supp. 2d`
becomes `fsuppd`, `N.Y.2d` becomes `nyd` -- and compared against the gazetteer
reduced the same way, with a similarity threshold rather than equality. Series
digits disappear in that reduction, which is deliberate: it is what lets
`F.3d` and `F.2d` collapse onto one key and a mangled `F,3ci` still land near
it.

Nothing here decides anything. It is a net, cast wide on purpose, and the
number it returns is how much a human or a model would have to read to be sure
the corpus holds no citation we have missed. Read against site hunting, the
interesting quantity is what this finds that hunting does not.

    uv run python -m exploration.locator_recall.fuzzy_sites
    uv run python -m exploration.locator_recall.fuzzy_sites --threshold 0.8 --show 40
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import io
import json
import re
from collections import Counter
from pathlib import Path

from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.adjudication import suspected_locators

BODY_MARKER = "--- Plain text ---\n"

# A number, a run of separators, some letters, a run of separators, a number.
# The letter run may carry its own punctuation and spaces -- that is what makes
# `F. Supp. 2d` one string rather than three -- but it may not carry a digit,
# so the two numbers cannot swallow the middle. Lengths are bounded so a match
# stays the size of a citation rather than a paragraph.
SITE = re.compile(
    r"(?<!\d)(\d{1,4})"
    r"([^0-9A-Za-z]{0,4})"
    r"([A-Za-z][A-Za-z.,'’’ \-&]{0,22}[A-Za-z.])"
    r"([^0-9A-Za-z]{0,4})"
    r"(\d{1,6})(?!\d)"
)
_LETTERS = re.compile(r"[^a-z]")

# Shapes that match number-string-number without being a locator. They are
# labelled rather than dropped: a filter would hide whatever it got wrong, and
# the point of the net is to be able to say what is left after everything
# explicable is explained.
_YEAR = re.compile(r"^(1[6-9]|20)\d\d$")
_MONTHS = frozenset(
    "january february march april may june july august september october november december "
    "jan feb mar apr jun jul aug sept sep oct nov dec".split()
)


def shape(text: str, start: int, end: int, match: re.Match[str]) -> str:
    """Why this number-string-number is not a citation, when it plainly is not."""
    before = text[max(0, start - 2) : start]
    after = text[end : end + 2]
    letters = letters_only(match.group(3))
    trailing_is_year = bool(_YEAR.match(match.group(5)))

    if letters in _MONTHS:
        return "a date"
    if trailing_is_year and ("(" in before or ")" in after or "(" in match.group(2)):
        return "a court parenthetical"
    if trailing_is_year:
        return "ends in a year"
    if "at" in match.group(4).lower() or text[max(0, start - 4) : start].lower().endswith("at "):
        return "a short-form pin cite"
    if letters in {"usc", "usca", "cfr", "uscs"} or "§" in text[end : end + 4]:
        return "a statute or regulation"
    # `529 F.3d at 935` matches as `529 F.` + `3`: the series digit inside the
    # reporter is a number, so the net closes early and reports a page of 3.
    # Anything immediately followed by a series letter is that, not a locator.
    if re.match(r"^(d|th|st|nd|rd)\b", text[end : end + 3]):
        return "cut short inside the reporter"
    if re.match(r"^\s*(at|\u00a7)\b", text[end : end + 6]):
        return "a short-form pin cite"
    if "@" in text[max(0, start - 30) : end + 30] or "Suite" in text[max(0, start - 40) : end + 10]:
        return "an address or contact block"
    return "unexplained"


def letters_only(text: str) -> str:
    """The string reduced to its letters, lowercased."""
    return _LETTERS.sub("", text.lower())


def gazetteer() -> dict[str, str]:
    """Every reporter eyecite knows, keyed by its letters alone.

    Editions rather than the reporters index, because that is the level at
    which a citation names one: `F.2d` and `F.3d` are separate editions of the
    Federal Reporter, and both reduce to `f`.
    """
    from eyecite.tokenizers import EDITIONS_LOOKUP

    keys: dict[str, str] = {}
    for spelling in EDITIONS_LOOKUP:
        key = letters_only(spelling)
        if key:
            keys.setdefault(key, spelling)
    return keys


def body(path: Path) -> str:
    _, marker, text = path.read_text(encoding="utf-8").partition(BODY_MARKER)
    return text if marker else path.read_text(encoding="utf-8")


def main() -> int:
    """Cast the net, and say what it holds that the other methods do not."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=Path("data/false-citation-bench-locator-only-v2.0"))
    parser.add_argument("--threshold", type=float, default=0.67)
    parser.add_argument(
        "--fuzzy-min-letters",
        type=int,
        default=4,
        help=(
            "shorter keys than this must match a reporter exactly. A two-letter key "
            "has no room for a typo to be told from a different word: at 67%%, `and` "
            "is 86%% similar to `Rand.`, and refusing to fuzz short keys is what "
            "keeps the net from filling with prose. Raising the length floor instead "
            "would be worse -- `U.S.` reduces to `us` and `F.` to `f`, so a floor of "
            "three loses most of the real reporters in the corpus."
        ),
    )
    parser.add_argument("--show", type=int, default=25)
    args = parser.parse_args()

    known = gazetteer()
    keys = list(known)
    print(f"gazetteer: {len(keys)} distinct reporter spellings once reduced to letters\n")

    records = [
        json.loads(line)
        for line in (args.bench / "extraction.jsonl").read_text().splitlines()
        if line.strip()
    ]
    truth: dict[str, list[tuple[int, int]]] = {}
    for record in records:
        truth.setdefault(record["document"], []).append((record["span"]["start"], record["span"]["end"]))

    totals = Counter()
    shapes: Counter = Counter()
    novel: list[tuple[str, str, str, float, str]] = []

    for path in sorted((args.bench / "documents_txt").glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
            hunted = suspected_locators(document)

        hunted_spans = [(site.span_start, site.span_end) for site in hunted]
        covered = truth.get(path.name, [])

        for match in SITE.finditer(text):
            totals["sites"] += 1
            key = letters_only(match.group(3))
            if not key:
                continue
            if len(key) < args.fuzzy_min_letters:
                close = [key] if key in known else []
            else:
                close = difflib.get_close_matches(key, keys, n=1, cutoff=args.threshold)
            if not close:
                continue
            totals["reporter-like"] += 1
            score = difflib.SequenceMatcher(None, key, close[0]).ratio()
            exact = key == close[0]
            totals["exact" if exact else "fuzzy"] += 1

            start, end = match.start(), match.end()
            if any(start < b and a < end for a, b in covered):
                totals["already ground truth"] += 1
                continue
            if any(start < b and a < end for a, b in hunted_spans):
                totals["also found by site hunting"] += 1
                continue
            totals["NEW"] += 1
            label = shape(text, start, end, match)
            shapes[label] += 1
            if label != "unexplained":
                continue
            window = " ".join(text[max(0, start - 60) : end + 60].split())
            novel.append((path.stem[:20], match.group(), known[close[0]], score, window))

    print("| bucket | count |")
    print("|---|---:|")
    for label in (
        "sites",
        "reporter-like",
        "exact",
        "fuzzy",
        "already ground truth",
        "also found by site hunting",
        "NEW",
    ):
        print(f"| {label} | {totals[label]} |")

    print("\n| what the new sites are | count |")
    print("|---|---:|")
    for label, count in shapes.most_common():
        print(f"| {label} | {count} |")

    print(f"\n--- {len(novel)} unexplained, showing {args.show} ---")
    for stem, matched, reporter, score, window in novel[: args.show]:
        print(f"  [{stem:<20}] {matched!r:<26} ~ {reporter!r} ({score:.0%})")
        print(f"      {window[:130]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
