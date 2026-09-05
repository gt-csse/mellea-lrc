"""Every secondary citation and the authority the tree attributed it to, for reading.

The attribution rate says how many landed, not whether they landed correctly. A
short form resolved to the wrong authority is worse than one left stranded: it
attaches a page claim to a case the filing never made it about, and no count
distinguishes the two.

So this prints each one to be read: the back-reference, the text around it, and
the full citation it was attributed to. Nothing is judged here.

    uv run python -m exploration.locator_recall.review_associations
    uv run python -m exploration.locator_recall.review_associations --document 006
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

from exploration.locator_recall.fuzzy_sites import body
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.structure.citation_tree import build_citation_tree

SECONDARY = frozenset({"ShortCaseCitation", "SupraCitation", "IdCitation", "ReferenceCitation"})
CONTEXT = 110


def kind(citation) -> str:
    return type(citation.citation).__name__


def main() -> int:
    """Print each attributed secondary citation beside its authority."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path)
    parser.add_argument("--document", help="only documents whose name contains this")
    args = parser.parse_args()

    documents = args.documents or Path("data/extraction-v2.0/documents_txt")
    total = 0

    for path in sorted(documents.glob("*.txt")):
        if args.document and args.document not in path.stem:
            continue
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
            tree = build_citation_tree(document)

        printed = False
        for authority in tree.authorities:
            secondary = [
                occurrence
                for occurrence in authority.occurrences
                if not occurrence.is_root and kind(occurrence.citation) in SECONDARY
            ]
            if not secondary:
                continue
            if not printed:
                print(f"\n########## {path.stem[:60]} ##########")
                printed = True
            root = authority.root
            print(f"\n  AUTHORITY  {' '.join(root.matched_text.split())!r}")
            for occurrence in secondary:
                total += 1
                citation = occurrence.citation
                start = citation.full_span.start
                window = " ".join(text[max(0, start - CONTEXT) : citation.full_span.end + 40].split())
                print(
                    f"    [{total:>3}] {kind(citation):<18}"
                    f"{' '.join(citation.matched_text.split())!r:<22} depth={occurrence.depth}"
                )
                print(f"          ...{window[-150:]!r}")

    print(f"\n{total} attributed secondary citations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
