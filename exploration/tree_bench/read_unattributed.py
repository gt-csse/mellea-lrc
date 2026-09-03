"""The citations the tree could not attribute, and the out-of-scope kinds.

Ground truth has to say what each of these is, so each is printed with enough
context to decide: a case citation whose full form is elsewhere, a reference
into the record, or not a citation at all.

    uv run python -m exploration.tree_bench.read_unattributed
"""

from __future__ import annotations

import argparse
import contextlib
import io
from collections import Counter
from pathlib import Path

from exploration.tree_bench.census import DOCS, body
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.structure.citation_tree import build_citation_tree

BEFORE, AFTER = 150, 80


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=DOCS)
    args = parser.parse_args()
    kinds = Counter()
    n = 0
    for path in sorted(args.docs.glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
        tree = build_citation_tree(document)
        for c in tree.out_of_scope:
            kinds[c.citation.kind.value] += 1
        for c in tree.unattributed:
            n += 1
            lo = max(0, c.full_span.start - BEFORE)
            print(
                f"\n[{n:>3}] {path.stem[:14]} @{c.full_span.start} {c.citation.kind.value} resolves_to={c.resolves_to}"
            )
            print(f"      ...{' '.join(text[lo : c.full_span.start].split())[-BEFORE:]}")
            print(f"      >>>{text[c.full_span.start : c.full_span.end]}<<<")
            print(f"      {' '.join(text[c.full_span.end : c.full_span.end + AFTER].split())}")
    print(f"\n{n} unattributed")
    print("\nout of scope, by kind")
    for k, v in kinds.most_common():
        print(f"  {k:<22}{v:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
