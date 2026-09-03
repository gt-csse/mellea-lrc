r"""The attribution failure rate on the mined corpus, once scope is decided.

Before the 154 were read the rate was a band from 0.3% to 6.1%, because nobody
had said whether an unattributed `Id.` referred to a case or to the record.
`scope_annotation` settles 146 of them and leaves 8 uncertain, which is narrow
enough to quote.

    uv run python -m exploration.generality.attribution_rate
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

from exploration.generality.scope_annotation import IN_SCOPE, OUT_OF_SCOPE, UNCERTAIN
from exploration.generality.survey import MINED, body
from mellea_lrc.core.citations import CitationKind
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.citation_tree import build_citation_tree

SECONDARY = frozenset({CitationKind.SHORT_CASE, CitationKind.ID, CitationKind.SUPRA, CitationKind.REFERENCE})

# The bench is the only annotated corpus, and there one attribution in 61
# secondary returns was moved off the tree's answer by hand. One observation, so
# the interval matters as much as the point: the 95% upper bound for 1 of 61.
BENCH_MISATTRIBUTION = 1 / 61
BENCH_MISATTRIBUTION_UPPER = 0.088


def main() -> int:
    """Print the rate, with the band the eight uncertain readings leave."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=MINED)
    args = parser.parse_args()

    attributed = secondary = 0
    for path in sorted(args.documents.glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
        for authority in build_citation_tree(document).authorities:
            for occurrence in authority.occurrences:
                attributed += 1
                if occurrence.citation.citation.kind in SECONDARY:
                    secondary += 1

    low_unattributed = len(IN_SCOPE)
    high_unattributed = len(IN_SCOPE) + len(UNCERTAIN)
    misattributed_low = BENCH_MISATTRIBUTION * secondary
    misattributed_high = BENCH_MISATTRIBUTION_UPPER * secondary

    print(f"  attributed occurrences        {attributed:>6}   ({secondary} of them secondary)")
    print(f"  unattributed and in scope     {low_unattributed:>6} to {high_unattributed}")
    print(f"  unattributed and out of scope {len(OUT_OF_SCOPE):>6}")
    print()
    for label, failures, extra in (
        ("lower", low_unattributed, misattributed_low),
        ("upper", high_unattributed, misattributed_high),
    ):
        denominator = attributed + (low_unattributed if label == "lower" else high_unattributed)
        rate = (failures + extra) / denominator
        print(
            f"  {label} bound: ({failures} unattributed + {extra:.0f} misattributed)"
            f" / {denominator} = {rate:.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
