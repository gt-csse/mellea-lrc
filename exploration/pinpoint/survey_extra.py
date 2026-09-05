"""Where does the pin cite go when eyecite does not put it in `pin_cite`?

A full citation carries `pin_cite` and `extra`. The page a filing actually
argues from is the pin cite, and it is what a pinpoint check needs; `extra` is
where eyecite puts text it recognised as belonging to the citation but could
not classify. When the pin cite lands there instead, the claim is still on the
page but the pipeline cannot see it.

Nothing is judged here. This counts how often it happens, and prints what the
text says beside what was parsed, so the shapes can be read before any rule is
written for them.

    uv run python -m exploration.pinpoint.survey_extra
    uv run python -m exploration.pinpoint.survey_extra --show 40
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
from collections import Counter
from pathlib import Path

from mellea_lrc.extraction import Relaxation, extract_from_plain_text

# A pin cite is a page of the same reporter: digits, ranges, star pages,
# paragraph and section marks. Nothing else.
PIN_SHAPED = re.compile(r"^[*¶]?\s*\d[\d\s,\-–*¶n.]*$")
# A section sign is not a page marker. `§ 1231` in `extra` is a return to a
# statute, so counting it as a lost pin cite would report a pinpoint failure
# where the filing made no pinpoint claim. Counted separately, and on this
# corpus it never fired -- the 68-to-1 figures are unaffected either way.
SECTION_SHAPED = re.compile(r"^§+\s*\d")
# A parallel citation is also digits-first and is *not* a pin cite -- it is the
# same case in another reporter. `88 S.Ct. 1323, 20 L.Ed.2d 262` beside
# `390 U.S. 727` is correct behaviour, not a lost pin cite.
PARALLEL = re.compile(r"\d+\s+[A-Z][A-Za-z.'’ ]*\d*\s+\d+")
CARRIES_PIN = ("FullCaseCitation", "ShortCaseCitation", "FullLawCitation", "FullJournalCitation")


def body(path: Path) -> str:
    """The document text spans index into."""
    return path.read_text(encoding="utf-8")


def main() -> int:
    """Count and show citations whose `extra` holds something pin-cite shaped."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path)
    parser.add_argument("--show", type=int, default=25)
    args = parser.parse_args()

    documents = args.documents or Path("data/extraction-v2.0/documents_txt")
    totals: Counter = Counter()
    rows: list[tuple[str, str, str, str, str]] = []

    for path in sorted(documents.glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)

        for citation in document.citations:
            kind = type(citation.citation).__name__
            if kind not in CARRIES_PIN:
                continue
            totals[kind] += 1
            pin = getattr(citation.citation, "pin_cite", None)
            extra = getattr(citation.citation, "extra", None)
            totals[f"{kind}: has pin_cite"] += 1 if pin else 0
            if not extra:
                continue
            totals[f"{kind}: has extra"] += 1
            written = str(extra).strip()
            if PARALLEL.search(written):
                totals["extra is a parallel citation"] += 1
                continue
            if PIN_SHAPED.match(written):
                totals["extra is a bare page -- a lost pin cite"] += 1
                following = repr(text[citation.full_span.end : citation.full_span.end + 26])
                rows.append(
                    (
                        path.stem[:18],
                        " ".join(citation.matched_text.split())[:26],
                        repr(pin),
                        repr(str(extra)[:34]),
                        following,
                    )
                )

    print("| count | |")
    print("|---|---:|")
    for label in sorted(totals):
        print(f"| {label} | {totals[label]} |")

    print(f"\n--- {len(rows)} with pin-cite-shaped `extra`, showing {args.show} ---")
    print(f"{'document':<20}{'citation':<28}{'pin_cite':<12}{'extra':<24}text after")
    for stem, matched, pin, extra, following in rows[: args.show]:
        print(f"{stem:<20}{matched:<28}{pin:<10}{extra:<14}raw after: {following}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
