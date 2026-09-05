"""Is the margin-interrupted `214 F.3d 1058` in the ground truth, or only the index one?

Document 022 states that citation twice: once in its table of authorities,
where it reads cleanly, and once in the argument, where the pleading-paper
margin falls between the reporter and the page. The second is the citation the
whole margin problem was found on. If only the first is in the denominator then
an extractor that never finds the second still scores full recall.

    uv run python -m exploration.margin_rules.probe_214
"""

from __future__ import annotations

import json
import re
from pathlib import Path

BENCH = Path("data/false-citation-bench-locator-only-v1.0")
PUBLISHED = Path("data/false-citation-bench/derived/extraction.jsonl")


def body(path: Path) -> str:
    """The document text spans index into."""
    return path.read_text(encoding="utf-8")


def main() -> None:
    document = next((BENCH / "documents_txt").glob("022*.txt"))
    text = body(document)

    print("=== records in locator-only-v1.0 for document 022 with volume 214 ===")
    for line in (BENCH / "extraction.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["document"].startswith("022") and record.get("volume") == "214":
            print(f"  {record['span']} {record['matched_text']!r}")

    print("\n=== records in the published bench, any document, volume 214 ===")
    for line in PUBLISHED.read_text().splitlines():
        record = json.loads(line)
        if record.get("volume") == "214":
            print(f"  {record['document'][:26]} {record['span']} {record['matched_text']!r}")

    print("\n=== every '214 F.3d' in the v1.1 text of document 022 ===")
    for hit in re.finditer(r"214\s*F\.\s*3d", text):
        window = text[hit.start() : hit.start() + 150]
        print(f"  at {hit.start()}: {window!r}")

    print("\n=== every '1058' in that text ===")
    for hit in re.finditer(r"\b1058\b", text):
        print(f"  at {hit.start()}: {text[max(0, hit.start() - 70) : hit.start() + 40]!r}")


if __name__ == "__main__":
    main()
