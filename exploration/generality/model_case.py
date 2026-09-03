r"""Where do the rules run out, on documents they were not built from?

The extraction rules generalise (see `survey`). This asks the next question:
what is left that a rule cannot reach, and is any of it worth a model call?

Three candidate pools, measured on the 77 mined filings and compared with the
26-document bench.

**Case names.** 17% of case citations carry no complete party pair, and some of
the ones that do are silently wrong -- the plaintiff filed as the defendant.
`exploration.court_and_date.probe_case_names` shows the three causes, none of
which is a separator that could be widened.

**Attribution of secondary citations.** Every `Id.`, short form and supra has to
be traced to the authority it means. eyecite resolves positionally, so it fails
where a reference was never extracted or the nearest citation is not the meant
one. On the bench this is 18 of 658 occurrences; on the mined corpus it is 252
of 2,702 -- three times the rate, and 210 of them bare `Id.`

**Site hunting.** Reporter strings that look like locators and produced no
citation. This is the pool a model was originally meant to adjudicate, and the
measurement argues against it: most candidates are mailing addresses.

    uv run python -m exploration.generality.model_case
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
from collections import Counter
from pathlib import Path

from exploration.generality.survey import BENCH, MINED, body
from mellea_lrc.adjudication import suspected_locators
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.citation_tree import build_citation_tree

# A letterhead, not a citation: `Corrales, NM 87048 (505) 220-5691`.
CONTACT = re.compile(r"\b\d{5}(?:-\d{4})?\b|\(?\d{3}\)?[ -]\d{3}-\d{4}|@|https?://|www\.")


def measure(directory: Path) -> Counter:
    """Count what a rule cannot settle in one corpus."""
    counts: Counter = Counter()
    for path in sorted(directory.glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)

        for item in document.citations:
            citation = item.citation
            if not isinstance(citation, FullCaseCitation):
                continue
            counts["case citations"] += 1
            if not (citation.plaintiff and citation.defendant):
                counts["an incomplete case name"] += 1

        tree = build_citation_tree(document)
        counts["occurrences attributed"] += tree.occurrence_count
        counts["secondary citations unattributed"] += len(tree.unattributed)

        for site in suspected_locators(document):
            counts["suspected locators"] += 1
            window = text[max(0, site.span_start - 60) : site.span_end + 60]
            if CONTACT.search(window):
                counts["  of those, a letterhead"] += 1
    return counts


def main() -> int:
    """Print both corpora side by side."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=BENCH)
    parser.add_argument("--mined", type=Path, default=MINED)
    args = parser.parse_args()

    bench, mined = measure(args.bench), measure(args.mined)
    print(f"{'':<36}{'bench':>10}{'mined':>10}")
    for label in (
        "case citations",
        "an incomplete case name",
        "occurrences attributed",
        "secondary citations unattributed",
        "suspected locators",
        "  of those, a letterhead",
    ):
        print(f"{label:<36}{bench[label]:>10}{mined[label]:>10}")

    for name, counts in (("bench", bench), ("mined", mined)):
        names = counts["an incomplete case name"] / max(counts["case citations"], 1)
        total = counts["occurrences attributed"] + counts["secondary citations unattributed"]
        attribution = counts["secondary citations unattributed"] / max(total, 1)
        print(
            f"\n{name}: {names:.1%} of case citations lack a complete name; "
            f"{attribution:.1%} of occurrences are unattributed"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
