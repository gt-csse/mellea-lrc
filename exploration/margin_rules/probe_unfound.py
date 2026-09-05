"""Why 18 bench locators are not findable verbatim in v1.1.

Prints each one with what stands at the same place in the newer text, so the
difference can be read rather than assumed.

    uv run python -m exploration.margin_rules.probe_unfound
"""

from __future__ import annotations

import re
from pathlib import Path

V11 = Path("data/corpus/renderings/v1.1")

PROBES = [
    ("008", "2010 WL4722279"),
    ("008", "2019 NYSlip Op 50388"),
    ("013", "455 US. 363"),
    ("022", "2016 WL1448829"),
    ("025", "2024 WL1076736"),
]


def body(stem: str) -> str:
    """The document text spans index into."""
    return next(V11.glob(f"{stem}*.txt")).read_text(encoding="utf-8")


def flexible(literal: str) -> re.Pattern[str]:
    """The same characters, with any whitespace allowed between every token."""
    return re.compile(r"\s*".join(re.escape(part) for part in literal.split()))


def main() -> None:
    for stem, literal in PROBES:
        text = body(stem)
        print(f"--- {stem}  {literal!r} ---")
        print(f"  verbatim:          {text.count(literal)}")

        loose = flexible(literal)
        hits = list(loose.finditer(text))
        print(f"  whitespace-loose:  {len(hits)}")
        for hit in hits[:2]:
            print(f"    {hit.group()!r} at {hit.start()}")

        # Also try it with the reporter's internal spacing free, for the
        # `WL1448829` and `NYSlip Op` shapes where a token itself split.
        squeezed = re.compile(r"\s*".join(re.escape(c) for c in literal.replace(" ", "")))
        squeezed_hits = list(squeezed.finditer(text))
        print(f"  character-loose:   {len(squeezed_hits)}")
        for hit in squeezed_hits[:2]:
            print(f"    {hit.group()!r} at {hit.start()}")
        print()


if __name__ == "__main__":
    main()
