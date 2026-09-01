"""Survey the published bench before rebuilding it as a locator-only ground truth.

Prints what kinds of record it holds, where the locators came from, and how
many of them can still be found verbatim in the v1.1 text -- which is the only
question that decides how hard re-anchoring will be.

    uv run python -m exploration.margin_rules.bench_survey
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

BENCH = Path("data/false-citation-bench/derived/extraction.jsonl")
V1 = Path("data/false-citation-bench/documents_txt")
V11 = Path("data/false-citation-bench-v1.1/documents_txt")
BODY_MARKER = "--- Plain text ---\n"


def body(path: Path) -> str:
    _, marker, text = path.read_text(encoding="utf-8").partition(BODY_MARKER)
    return text if marker else path.read_text(encoding="utf-8")


def main() -> None:
    records = [json.loads(line) for line in BENCH.read_text().splitlines() if line.strip()]
    print(f"records: {len(records)}")
    print(f"kinds:   {dict(Counter(r['kind'] for r in records))}")
    print(f"sources: {dict(Counter(r.get('source', '(none)') for r in records))}")
    print(f"fields:  {sorted({k for r in records for k in r})}")

    locators = [r for r in records if r["kind"] == "locator"]
    print(f"\nlocators: {len(locators)}, dockets: {len(records) - len(locators)}")

    exact = span_ok = ambiguous = absent = 0
    misses = []
    for record in locators:
        stem = record["document"].removesuffix(".txt")
        v11 = V11 / f"{stem}.txt"
        if not v11.exists():
            absent += 1
            continue
        text = body(v11)
        hits = text.count(record["matched_text"])
        if hits == 1:
            exact += 1
            if text.find(record["matched_text"]) == record["span"]["start"]:
                span_ok += 1
        elif hits > 1:
            ambiguous += 1
        else:
            misses.append(record)

    print(f"\nre-anchoring the {len(locators)} locators into v1.1 by verbatim text:")
    print(f"  found exactly once:          {exact}")
    print(f"    ... and at the same offset {span_ok}")
    print(f"  found more than once:        {ambiguous}")
    print(f"  not found verbatim:          {len(misses)}")
    print(f"  document absent:             {absent}")

    print("\n  not found verbatim -- these need reading:")
    for record in misses[:25]:
        print(f"    {record['document'][:26]:<28}{record['matched_text']!r:<28}{record.get('source', '')}")


if __name__ == "__main__":
    main()
