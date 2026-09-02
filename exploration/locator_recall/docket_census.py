"""How many filings cite a docket number, against how many the bench records.

The bench carries 11 docket occurrences in 4 documents, every one of them found
by the model-assisted docket hunt. That is what was found and accepted, not an
exhaustive annotation, so it is a floor. This sweeps the text for the shape a
federal docket number has and says how far above that floor the corpus sits.

The shape is deliberately strict -- `1:25-cv-05745`, an office number, a
two-digit year, a case-type code and a sequence -- because a loose pattern
matches phone numbers and statutory subsections. Anything it finds is a docket
number; what it misses is state and appellate forms that do not follow it.

    uv run python -m exploration.locator_recall.docket_census
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from exploration.locator_recall.fuzzy_sites import body

# office:year-type-sequence, with the office and the leading `No.` optional.
DOCKET = re.compile(
    r"(?:\bNo\.?\s*)?\b\d{1,2}:\d{2}-?\s?(?:cv|cr|bk|md|mj|mc|ap|civ)-?\s?\d{3,6}(?:-[A-Z]{2,4})*",
    re.IGNORECASE,
)


def main() -> int:
    """Count docket-shaped strings per document, and compare with the bench."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench", type=Path, default=Path("data/false-citation-bench-locator-only-v2.0")
    )
    parser.add_argument(
        "--published",
        type=Path,
        default=Path("data/false-citation-bench/derived/extraction.jsonl"),
    )
    args = parser.parse_args()

    recorded = Counter()
    for line in args.published.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["kind"] == "docket":
            recorded[record["document"][:3]] += 1

    # A filing's own docket number is not a citation. It stands in the caption
    # and in every ECF page stamp -- `Case 2:25-cv-01295-GMS Document 1 Filed
    # 04/18/25` -- which is why one document carries twenty of them. Those
    # stamps are page furniture Docling did not drop, so they are in the text
    # whether or not anyone wants them counted.
    caption = re.compile(r"Case\s+(No\.?:?\s*)?$", re.IGNORECASE)

    found: Counter = Counter()
    cited: Counter = Counter()
    examples: dict[str, list[str]] = {}
    for path in sorted((args.bench / "documents_txt").glob("*.txt")):
        text = body(path)
        own: set[str] = set()
        others: list[str] = []
        for match in DOCKET.finditer(text):
            written = " ".join(match.group().split())
            if caption.search(text[max(0, match.start() - 12) : match.start()]):
                own.add(re.sub(r"[^0-9a-z]", "", written.lower()))
            else:
                others.append(written)
        # The caption number is the same number wherever else it appears; only
        # a different one is a citation to another case.
        others = [o for o in others if re.sub(r"[^0-9a-z]", "", o.lower()) not in own]
        stem = path.stem[:3]
        found[stem] = len(own) + len(others)
        cited[stem] = len(others)
        examples[stem] = sorted(set(others))[:3]

    print(f"{'doc':<6}{'bench':>7}{'cites another docket':>22}  examples")
    for stem in sorted(set(found) | set(recorded)):
        if not cited.get(stem) and not recorded.get(stem):
            continue
        print(f"{stem:<6}{recorded.get(stem, 0):>7}{cited.get(stem, 0):>22}  {examples.get(stem, [])}")

    citing = {stem for stem, count in cited.items() if count}
    holding = {stem for stem, count in found.items() if count}
    print(f"\ndocuments the bench records a docket citation in: {len(recorded)} of 26")
    print(f"documents citing another case's docket:           {len(citing)} of 26")
    print(f"documents holding any docket number at all:       {len(holding)} of 26")
    print(
        f"occurrences: bench {sum(recorded.values())}, "
        f"cited {sum(cited.values())}, all {sum(found.values())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
