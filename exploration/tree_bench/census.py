"""What the tree holds on locator-only-v2.0, before any annotation decisions.

Counts authorities, occurrences by kind and depth, and the two reported
failure classes, so the shape of the ground truth is known before it is built.

    uv run python -m exploration.tree_bench.census
"""

from __future__ import annotations

import contextlib
import io
from collections import Counter
from pathlib import Path

from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.structure.citation_tree import build_citation_tree

BODY_MARKER = "--- Plain text ---\n"
DOCS = Path("data/false-citation-bench-locator-only-v2.0/documents_txt")


def body(path: Path) -> str:
    _, marker, rest = path.read_text(encoding="utf-8").partition(BODY_MARKER)
    return rest if marker else path.read_text(encoding="utf-8")


def main() -> int:
    kinds, depths, totals = Counter(), Counter(), Counter()
    for path in sorted(DOCS.glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
        tree = build_citation_tree(document)
        totals["authorities"] += len(tree.authorities)
        totals["occurrences"] += tree.occurrence_count
        totals["unattributed"] += len(tree.unattributed)
        totals["out_of_scope"] += len(tree.out_of_scope)
        totals["pinpoint claims"] += tree.pinpoint_claim_count
        for authority in tree.authorities:
            for occurrence in authority.occurrences:
                kinds[occurrence.citation.citation.kind.value] += 1
                depths[occurrence.depth] += 1
    print("totals")
    for k, v in totals.items():
        print(f"  {k:<18}{v:>6}")
    print("\nattributed occurrences by kind")
    for k, v in kinds.most_common():
        print(f"  {k:<22}{v:>6}")
    print("\nby depth (0 = the full citation that introduced the authority)")
    for d in sorted(depths):
        print(f"  depth {d:<16}{depths[d]:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
