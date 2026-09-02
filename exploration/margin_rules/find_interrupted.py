"""Find locators the ground truth is missing because a margin interrupts them.

A citation whose page is separated from its reporter by a column of line
numbers is still stated by the filing. The published bench left those out --
its own README calls them occurrences where "the filing states no complete
identifier" -- and with them out of the denominator, an extractor that never
finds them still scores full recall.

They are discoverable because we hold the same documents twice. In v2.0 the
margin is gone. The paragraph breaks it sat between are not, so the page still
stands past a blank line and only FULL reads it -- BOUNDED on v2.0 finds
nothing extra, which is why the first attempt at this reported none. In v1.1
the margin is still there. Anything FULL finds on v2.0 that the bench does not
carry is a candidate, and the gutter pattern below is what confirms it against
the v1.1 text rather than trusting the tokenizer.

    uv run python -m exploration.margin_rules.find_interrupted
"""

from __future__ import annotations

import contextlib
import io
import json
import re
from collections import Counter
from pathlib import Path

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction import Relaxation, extract_from_plain_text

BENCH = Path("data/false-citation-bench-locator-only-v1.0")
CLEAN = Path("data/false-citation-bench-v2.0/documents_txt")
BODY_MARKER = "--- Plain text ---\n"


def body(path: Path) -> str:
    _, marker, text = path.read_text(encoding="utf-8").partition(BODY_MARKER)
    return text if marker else path.read_text(encoding="utf-8")


def key(volume: str, reporter: str, page: str) -> str:
    return f"{volume}|{''.join(str(reporter).split())}|{page}"


def gutter_pattern(volume: str, reporter: str, page: str) -> re.Pattern[str]:
    """Volume, reporter, then a run of line numbers, then the page.

    The line numbers are what stands between the reporter and its page, so the
    pattern names them explicitly rather than allowing arbitrary text: only
    whitespace and standalone integers may intervene, which is what a margin
    is and what prose is not.
    """
    parts = [re.escape(volume), r"\s*"]
    parts += [r"\s*".join(re.escape(c) for c in reporter if not c.isspace())]
    parts += [r"(?:\s+\d{1,3})+\s+", re.escape(page)]
    return re.compile("".join(parts))


def main() -> None:
    records = [
        json.loads(line) for line in (BENCH / "extraction.jsonl").read_text().splitlines() if line.strip()
    ]
    expected: dict[str, Counter] = {}
    for record in records:
        expected.setdefault(record["document"], Counter())[
            key(record["volume"], record["reporter"], record["page"])
        ] += 1

    total = 0
    for path in sorted(CLEAN.glob("*.txt")):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(body(path), relaxation=Relaxation.FULL)
        found: Counter = Counter()
        for citation in document.citations:
            inner = citation.citation
            if isinstance(inner, FullCaseCitation) and inner.volume and inner.reporter and inner.page:
                found[key(inner.volume, inner.reporter, inner.page)] += 1

        have = expected.get(path.name, Counter())
        missing = {k: found[k] - have.get(k, 0) for k in found if found[k] > have.get(k, 0)}
        if not missing:
            continue

        v11 = body(next((BENCH / "documents_txt").glob(f"{path.name[:3]}*.txt")))
        print(f"--- {path.stem[:52]} ---")
        for identifier, count in missing.items():
            volume, reporter, page = identifier.split("|")
            hits = list(gutter_pattern(volume, reporter, page).finditer(v11))
            total += count
            print(
                f"  {identifier:<24} v2.0 has {count} more than the bench; gutter hits in v1.1: {len(hits)}"
            )
            for hit in hits:
                shown = hit.group()[:38].replace("\n", "\\n")
                print(f"      {hit.start()}-{hit.end()}  {shown!r}...")
    print(f"\ncitations the bench is missing: {total}")


if __name__ == "__main__":
    main()
