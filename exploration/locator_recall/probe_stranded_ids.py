"""What the unattributed `Id.` citations are actually pointing at.

The citation-tree handoff guesses: "seventeen carry a paragraph pin cite and are
probably references into a pleading's own numbered allegations". That is a
guess, and it decides something real -- whether these are a tree failure worth
fixing or a filing referring to itself, which no citation tree should attribute
to an authority at all.

This prints each one with what precedes it, which is the only thing an `Id.`
can be pointing at.

    uv run python -m exploration.locator_recall.probe_stranded_ids
"""

from __future__ import annotations

import argparse
import contextlib
import io
from collections import Counter
from pathlib import Path

from exploration.locator_recall.fuzzy_sites import body
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.structure.citation_tree import build_citation_tree

BEFORE = 240


def main() -> int:
    """Print every unattributed back-reference with the text that precedes it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path)
    parser.add_argument("--show", type=int, default=20)
    args = parser.parse_args()

    documents = args.documents or Path("data/extraction-v2.0/documents_txt")
    shown = 0
    per_document: Counter = Counter()

    for path in sorted(documents.glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
            tree = build_citation_tree(document)
        if not tree.unattributed:
            continue

        per_document[path.stem[:26]] = len(tree.unattributed)
        print(f"=== {path.stem[:56]} — {len(tree.unattributed)} unattributed ===")
        for citation in tree.unattributed:
            if shown >= args.show:
                break
            shown += 1
            start = citation.full_span.start
            preceding = " ".join(text[max(0, start - BEFORE) : start].split())
            pin = getattr(citation.citation, "pin_cite", None)
            print(f"  {type(citation.citation).__name__} {citation.matched_text!r}  pin={pin!r}")
            print(f"    ...{preceding[-200:]!r}")
        print()

    print(f"unattributed per document: {dict(per_document)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
