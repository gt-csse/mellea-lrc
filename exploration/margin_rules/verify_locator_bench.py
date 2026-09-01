"""Check that every span in the locator-only bench points at what it claims.

A bench with a wrong span is worse than one with a gap, so this reads each
record's span out of the text it is anchored to and compares.

    uv run python -m exploration.margin_rules.verify_locator_bench
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path

BENCH = Path(sys.argv[1] if len(sys.argv) > 1 else "data/false-citation-bench-locator-only-v1.0")
BODY_MARKER = "--- Plain text ---\n"


def body(path: Path) -> str:
    _, marker, text = path.read_text(encoding="utf-8").partition(BODY_MARKER)
    return text if marker else path.read_text(encoding="utf-8")


def main() -> None:
    records = [
        json.loads(line) for line in (BENCH / "extraction.jsonl").read_text().splitlines() if line.strip()
    ]
    texts = {path.name: body(path) for path in (BENCH / "documents_txt").glob("*.txt")}

    print(f"records: {len(records)}")
    print(f"kinds:   {dict(Counter(r['kind'] for r in records))}")
    print(f"docs:    {len({r['document'] for r in records})} of {len(texts)}")

    wrong = []
    for record in records:
        text = texts[record["document"]]
        actual = text[record["span"]["start"] : record["span"]["end"]]
        if actual != record["matched_text"]:
            wrong.append((record, actual))

    print(f"\nspans that read back exactly: {len(records) - len(wrong)} of {len(records)}")
    for record, actual in wrong[:10]:
        print(f"  {record['document'][:24]:<26}claims {record['matched_text']!r} reads {actual!r}")

    noted = [r for r in records if r.get("anchor_note")]
    print(f"\nanchored on the locator rather than the text: {len(noted)}")
    for record in noted:
        print(
            f"  {record['document'][:24]:<26}{record['matched_text']!r}  ({record['volume']}|{record['reporter']}|{record['page']})"
        )

    overlaps = 0
    for document in {r["document"] for r in records}:
        spans = sorted((r["span"]["start"], r["span"]["end"]) for r in records if r["document"] == document)
        overlaps += sum(1 for a, b in pairwise(spans) if a[1] > b[0])
    print(f"\noverlapping spans within a document: {overlaps}")


if __name__ == "__main__":
    main()
