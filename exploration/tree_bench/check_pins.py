"""Look for pin cites the reader may have invented.

The dangerous false positive is a parallel citation: `390 U.S. 727, 88 S.Ct.
1323` reads as page 88 unless something stops it. This prints every recorded
pin cite whose following text looks like a reporter, plus the rare shapes, with
their context, so each can be judged against the document.

    uv run python -m exploration.tree_bench.check_pins
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from exploration.tree_bench.census import body

BENCH = Path("data/false-citation-bench-tree-v2.0")
# A reporter directly after the supposed pin cite: the pin cite is a volume.
LOOKS_LIKE_REPORTER = re.compile(r"^[^\S\r\n]*[A-Z][A-Za-z.’'’]*\s*\d*[a-z.]*\s+\d+")
RARE = re.compile(r"[§,n]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=BENCH)
    args = parser.parse_args()
    texts = {p.name: body(p) for p in (args.bench / "documents_txt").glob("*.txt")}
    suspect = rare = 0
    for line in (args.bench / "occurrences.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        pin = record["pin_cite_written"]
        if not pin:
            continue
        text = texts[record["document"]]
        after = text[pin["span"]["end"] : pin["span"]["end"] + 40]
        flag = ""
        if LOOKS_LIKE_REPORTER.match(after):
            flag, suspect = "PARALLEL?", suspect + 1
        elif RARE.search(pin["text"]):
            flag, rare = "rare", rare + 1
        if flag:
            start = record["locator_span"]["start"]
            print(
                f"{flag:<10}{record['document'][:12]:<14}{record['kind'][:14]:<16}"
                f"{text[start : record['locator_span']['end']]!r} pin={pin['text']!r} "
                f"after={' '.join(after.split())[:36]!r}"
            )
    print(f"\n{suspect} suspected parallel reads, {rare} rare shapes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
