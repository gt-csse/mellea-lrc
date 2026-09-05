"""Find what became of `455 US. 363` in document 013 across the three versions.

It is the bench's one hand-added locator -- a reporter written without the
period after US, which no gazetteer reaches -- and it is the case that decides
what a locator-only ground truth has to be anchored on.

    uv run python -m exploration.margin_rules.probe_455
"""

from __future__ import annotations

import re
from pathlib import Path

VERSIONS = {
    "v1": Path("data/false-citation-bench/documents_txt"),
    "v1.1": Path("data/corpus/renderings/v1.1"),
    "v2.0": Path("data/extraction-v2.0/documents_txt"),
}
NEEDLE = re.compile(r"455[^\n]{0,12}?\b(?:US|U\.\s?S)\.?[^\n]{0,12}?363|455\s*U\s*\.?\s*S\.?\s*363")


def body(directory: Path, stem: str) -> str:
    """The document text spans index into."""
    return next(directory.glob(f"{stem}*.txt")).read_text(encoding="utf-8")


def main() -> None:
    for label, directory in VERSIONS.items():
        text = body(directory, "013")
        print(f"--- {label} ---")
        hits = list(NEEDLE.finditer(text))
        print(f"  pattern hits: {len(hits)}")
        for hit in hits:
            print(f"    {hit.group()!r} at {hit.start()}")
        for position in [m.start() for m in re.finditer(r"\b455\b", text)]:
            print(f"    context at {position}: {text[position - 40 : position + 60]!r}")
        print()


if __name__ == "__main__":
    main()
