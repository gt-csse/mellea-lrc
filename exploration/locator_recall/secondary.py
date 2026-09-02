"""How much of a filing's secondary citation the tree actually accounts for.

A brief cites an authority once in full and then returns to it -- `Id. at 570`,
`550 U.S. at 563`, or by party name. Those return visits are the larger part of
what a filing asserts, and each names its own page and attaches its own
proposition, so a pipeline that only checks full citations checks one claim per
authority and skips the rest.

`build_citation_tree` resolves each return visit, transitively, to the full
citation that introduced the authority. This asks what fraction of them it
lands, which is the question that decides whether the tree is ready to sit
under the masking step: anything it cannot attribute is a citation the residue
hunt will keep offering to a model.

Reported per relaxation level, because the level decides which full citations
exist to be an antecedent at all.

    uv run python -m exploration.locator_recall.secondary
    uv run python -m exploration.locator_recall.secondary --documents <dir>
"""

from __future__ import annotations

import argparse
import contextlib
import io
from collections import Counter
from pathlib import Path

from exploration.locator_recall.fuzzy_sites import body
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.citation_tree import build_citation_tree

# The kinds that point back at an authority named earlier, rather than naming
# one themselves. `ReferenceCitation` is a party name standing in for the case.
SECONDARY = frozenset({"ShortCaseCitation", "SupraCitation", "IdCitation", "ReferenceCitation"})


def kind(citation) -> str:
    return type(citation.citation).__name__


def main() -> int:
    """Count secondary citations, and how many the tree attributes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path)
    args = parser.parse_args()

    documents = args.documents or Path("data/false-citation-bench-locator-only-v2.0/documents_txt")
    paths = sorted(documents.glob("*.txt"))
    if not paths:
        print(f"{documents}: no .txt documents found")
        return 1
    print(f"{len(paths)} documents from {documents}\n")

    for level in (Relaxation.BOUNDED, Relaxation.FULL):
        totals: Counter = Counter()
        attributed_kinds: Counter = Counter()
        stranded_kinds: Counter = Counter()
        pin_cites = 0

        for path in paths:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                document = extract_from_plain_text(body(path), relaxation=level)
                tree = build_citation_tree(document)

            totals["citations"] += len(document.citations)
            totals["secondary in the text"] += sum(1 for c in document.citations if kind(c) in SECONDARY)
            totals["authorities"] += len(tree.authorities)
            totals["out of scope"] += len(tree.out_of_scope)

            for authority in tree.authorities:
                for occurrence in authority.occurrences:
                    if occurrence.is_root:
                        continue
                    # A non-root occurrence is not the same thing as a secondary
                    # citation. A brief that states one case in full twice has a
                    # second full citation attributed to the same authority, and
                    # counting those as secondary was what produced a share above
                    # 100%.
                    totals["repeat occurrences"] += 1
                    if kind(occurrence.citation) in SECONDARY:
                        totals["secondary attributed"] += 1
                        attributed_kinds[kind(occurrence.citation)] += 1
                    else:
                        totals["repeat full citations"] += 1
                pin_cites += len(authority.pin_cites)

            for citation in tree.unattributed:
                stranded_kinds[kind(citation)] += 1
                totals["unattributed"] += 1
                if kind(citation) in SECONDARY:
                    totals["secondary unattributed"] += 1

            for citation in tree.out_of_scope:
                if kind(citation) in SECONDARY:
                    totals["secondary out of scope"] += 1

        secondary = totals["secondary in the text"]
        landed = totals["secondary attributed"]
        share = f"{landed / secondary:.1%}" if secondary else "-"
        accounted = landed + totals["secondary unattributed"] + totals["secondary out of scope"]

        print(f"=== {level.value} ===")
        print(f"  citations extracted           {totals['citations']}")
        print(f"  authorities                   {totals['authorities']}")
        print(f"  distinct pinpoint claims      {pin_cites}")
        print()
        print(f"  secondary citations in text   {secondary}")
        print(f"    attributed to an authority  {landed}   ({share})")
        print(f"    unattributed                {totals['secondary unattributed']}")
        print(f"    out of scope (not a case)   {totals['secondary out of scope']}")
        print(f"    accounted for               {accounted} of {secondary}")
        print()
        print(f"  repeat full citations         {totals['repeat full citations']}")
        print(f"  all repeat occurrences        {totals['repeat occurrences']}")
        print(f"  everything out of scope       {totals['out of scope']}")
        print(f"  attributed secondary by kind  {dict(attributed_kinds)}")
        print(f"  unattributed by kind          {dict(stranded_kinds)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
