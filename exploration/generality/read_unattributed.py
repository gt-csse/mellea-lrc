r"""Every unattributed occurrence on the mined corpus, with the context to judge it.

The tree cannot say whether a bare `Id.` points at a case or at the record --
only positive evidence sends a citation out of scope, and a bare `Id.` carries
none. That leaves the scope of these undecided, which is what makes the
attribution rate a band rather than a number.

This prints them for reading. Occurrences carrying a paragraph or page-and-line
pin cite are skipped by default: those already have their evidence.

    uv run python -m exploration.generality.read_unattributed
    uv run python -m exploration.generality.read_unattributed --all
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
from pathlib import Path

from exploration.generality.survey import MINED, body
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.citation_tree import build_citation_tree

RECORD_EVIDENCE = re.compile(r"\d+\s*:\s*\d+|¶")
BEFORE, AFTER = 150, 40


def main() -> int:
    """Print each undecided occurrence with the sentence around it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=MINED)
    parser.add_argument("--all", action="store_true", help="include the ones already evidenced")
    args = parser.parse_args()

    n = 0
    for path in sorted(args.documents.glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
        undecided = []
        for item in build_citation_tree(document).unattributed:
            window = text[item.span.start : item.span.end] + text[item.span.end : item.span.end + 22]
            if not args.all and RECORD_EVIDENCE.search(window):
                continue
            undecided.append(item)
        if not undecided:
            continue
        print(f"\n### {path.stem}  ({len(undecided)})")
        for item in undecided:
            n += 1
            start = max(0, item.span.start - BEFORE)
            before = " ".join(text[start : item.span.start].split())[-BEFORE:]
            after = " ".join(text[item.span.end : item.span.end + AFTER].split())
            print(f"[{n:>3}] {item.citation.kind.value[:9]:<10}{item.matched_text[:22]!r}")
            print(f"      ...{before}  >>>{item.matched_text}<<<  {after}")
    print(f"\n{n} undecided occurrences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
