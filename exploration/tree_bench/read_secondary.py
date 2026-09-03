"""Every secondary occurrence the tree attributed, with the context to judge it.

Ground truth records where a secondary citation *should* belong, so each one
has to be read against the authority the tree gave it rather than trusted.
Prints the authority, the citation, and the surrounding text.

    uv run python -m exploration.tree_bench.read_secondary
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

from exploration.tree_bench.census import DOCS, body
from mellea_lrc.core.citations import CitationKind
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.structure.citation_tree import build_citation_tree

SECONDARY = {CitationKind.SHORT_CASE, CitationKind.ID, CitationKind.SUPRA, CitationKind.REFERENCE}
BEFORE, AFTER = 180, 90


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=DOCS)
    args = parser.parse_args()
    n = 0
    for path in sorted(args.docs.glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
        tree = build_citation_tree(document)
        for authority in tree.authorities:
            root = authority.root.citation
            label = f"{root.volume} {root.reporter} {root.page}"
            for occ in authority.occurrences:
                if occ.citation.citation.kind not in SECONDARY:
                    continue
                n += 1
                c = occ.citation
                lo = max(0, c.span.start - BEFORE)
                print(
                    f"\n[{n:>3}] {path.stem[:14]} @{c.span.start} {c.citation.kind.value} depth={occ.depth}"
                )
                print(f"      authority: {label}  ({root.plaintiff} v. {root.defendant})")
                print(f"      parsed pin: {occ.pin_cite!r}")
                print(f"      ...{' '.join(text[lo : c.span.start].split())[-BEFORE:]}")
                print(f"      >>>{text[c.span.start : c.span.end]}<<<")
                print(f"      {' '.join(text[c.span.end : c.span.end + AFTER].split())}")
    print(f"\n{n} secondary occurrences attributed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
