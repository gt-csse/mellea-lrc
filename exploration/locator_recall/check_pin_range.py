"""Find attributions whose pin cite cannot be a page of the authority.

Reading all 70 attributed secondary citations by hand turned up one wrong, and
it is wrong in a way arithmetic can see. Document 022 says:

    The cited case, Doe v. Commonwealth's Attorney, 403 F. Supp. 1199
    (E.D. Va. 1975), is inapposite ... Moreover, Advanced Textile itself
    granted anonymity for civil labor claims ... Id. at 1072-73.

The sentence is about Advanced Textile, `214 F.3d 1058`. The nearest preceding
citation is `403 F. Supp. 1199`, and eyecite resolves `Id.` positionally, so
the pin cite lands on the wrong case. Page 1072 is 127 *below* that case's
first page, which no page of it can be, and it sits comfortably inside
Advanced Textile, whose other pin cites in the same filing are 1068 and
1071-72.

eyecite's own `_has_invalid_pin_cite` does not catch it: that test allows a pin
cite within 150 of the first page in either direction. Requiring the pin cite
to be at or after the first page is what separates them.

The check is deliberately one-sided. A pin cite far *above* the first page is
ordinary -- a long opinion runs for hundreds of pages -- but one *below* it is
impossible, so only the below case is reported.

It also applies only where the two numbers are the same numbering. A Westlaw or
LEXIS citation's page is a document number and its pin cite is a star page:
`2024 WL 1076736, at *6` is not a citation to page 6 of anything. Comparing
those flagged 36 perfectly good citations before the star-page test was added,
which is a fair warning about how such a rule reads on an unfamiliar corpus.

One more thing this taught: the pin cite has to be read from the text, not
from the parse. On the very case the check was written for, eyecite attributes
the `Id.` to the wrong authority *and* records `pin_cite=None`, having thrown
`at 1072 -73` away. A check that trusted the parse would see nothing to check.
A check on the parser's own output cannot see the parser's own mistake.

    uv run python -m exploration.locator_recall.check_pin_range
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
from pathlib import Path

from exploration.locator_recall.fuzzy_sites import body
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.structure.citation_tree import build_citation_tree

FIRST_NUMBER = re.compile(r"\d+")
# `Id. at 1072-73`, `, at 546`, `at *6`. Read off the text that follows the
# citation rather than off `pin_cite`, which the parser may have discarded.
WRITTEN_PIN = re.compile(r"^[\s,]*at\s+(\*?)\s*(\d[\d,\s\-–]*)")


def page_of(value: str | None) -> int | None:
    """The first number in a page or pin cite, if it has one."""
    if not value:
        return None
    match = FIRST_NUMBER.search(str(value))
    return int(match.group()) if match else None


def main() -> int:
    """Report every occurrence whose pin cite falls below its authority's page."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path)
    args = parser.parse_args()

    documents = args.documents or Path("data/extraction-v2.0/documents_txt")
    checked = flagged = 0

    for path in sorted(documents.glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
            tree = build_citation_tree(document)

        for authority in tree.authorities:
            reporter = str(getattr(authority.root.citation, "reporter", "") or "")
            if "WL" in reporter or "LEXIS" in reporter.upper():
                continue
            first = page_of(getattr(authority.root.citation, "page", None))
            if first is None:
                continue
            for occurrence in authority.occurrences:
                if occurrence.is_root:
                    continue
                # A short form carries its pin cite inside itself -- the `1154`
                # of `695 F.Supp.2d at 1154` is the citation's own page -- while
                # an `Id.` carries it in the text that follows. Both are the
                # same claim about the same numbering, so both are checked.
                inner = getattr(occurrence.citation.citation, "page", None)
                if type(occurrence.citation.citation).__name__ == "ShortCaseCitation" and inner:
                    written, pin = str(inner), page_of(inner)
                else:
                    after = text[occurrence.citation.full_span.end : occurrence.citation.full_span.end + 24]
                    match = WRITTEN_PIN.match(after)
                    if match is None or match.group(1):
                        continue
                    written, pin = match.group(2).strip(), page_of(match.group(2))
                if pin is None:
                    continue
                checked += 1
                if pin >= first:
                    continue
                flagged += 1
                start = occurrence.citation.full_span.start
                window = " ".join(text[max(0, start - 170) : occurrence.citation.full_span.end + 30].split())
                print(f"{path.stem[:30]}")
                print(f"  authority {' '.join(authority.root.matched_text.split())!r} first page {first}")
                print(
                    f"  occurrence {' '.join(occurrence.citation.matched_text.split())!r} "
                    f"written pin {written!r} -> page {pin}, {first - pin} below "
                    f"(the parser recorded {occurrence.pin_cite!r})"
                )
                print(f"  ...{window[-150:]!r}\n")

    print(f"occurrences with a numeric pin cite: {checked}")
    print(f"pin cite below the authority's first page: {flagged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
