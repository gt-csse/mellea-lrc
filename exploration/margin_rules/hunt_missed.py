"""Hunt the locator-only v2.0 corpus for citations the ground truth may still miss.

`suspected_locators` masks every citation the extractor found and sweeps what is
left for any reporter string the gazetteer knows with digits close on both
sides -- the volume-and-page shape. It is recall-oriented by design: most sites
are noise, and the point is that a real locator no tokenizer reaches cannot
hide from it, because the reporter is still spelled out on the page.

Sites are also checked against the bench, not just against the extractor, so a
citation the extractor finds but the ground truth omits shows up too.

No model runs. Adjudication is a separate step; this only says where to look.

    uv run python -m exploration.margin_rules.hunt_missed
"""

from __future__ import annotations

import contextlib
import io
import json
import re
from collections import Counter
from pathlib import Path

from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.adjudication import suspected_locators

BENCH = Path("data/extraction-v2.0")
BODY_MARKER = "--- Plain text ---\n"
WINDOW = 90


def body(path: Path) -> str:
    _, marker, text = path.read_text(encoding="utf-8").partition(BODY_MARKER)
    return text if marker else path.read_text(encoding="utf-8")


def main() -> None:
    records = [
        json.loads(line) for line in (BENCH / "extraction.jsonl").read_text().splitlines() if line.strip()
    ]
    truth: dict[str, list[tuple[int, int]]] = {}
    for record in records:
        truth.setdefault(record["document"], []).append((record["span"]["start"], record["span"]["end"]))

    total_sites = 0
    reporters: Counter = Counter()
    interesting: list[tuple[str, str, str]] = []

    for path in sorted((BENCH / "documents_txt").glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
            sites = suspected_locators(document)

        covered = truth.get(path.name, [])
        for site in sites:
            total_sites += 1
            reporters[site.reporter] += 1
            # A site sitting inside a bench record is a citation already known.
            if any(site.span_start < end and start < site.span_end for start, end in covered):
                continue
            start = max(0, site.span_start - WINDOW)
            window = " ".join(text[start : site.span_end + WINDOW].split())
            interesting.append((path.stem[:22], site.reporter, window))

    print(
        f"sites hunted: {total_sites} across {len(list((BENCH / 'documents_txt').glob('*.txt')))} documents"
    )
    print(f"reporters flagged: {dict(reporters.most_common(12))}")
    print(f"\nsites not inside any ground-truth span: {len(interesting)}\n")

    # A site is worth reading when digits sit close on both sides of the
    # reporter, which is the shape of a locator rather than of prose.
    shaped = re.compile(r"\d[\d,\s.]{0,12}$")
    for stem, reporter, window in interesting:
        head = window[: window.rfind(reporter)] if reporter in window else window
        mark = "  <-- digits before" if shaped.search(head[-14:]) else ""
        print(f"  [{stem:<22}] {reporter!r:<14}{mark}")
        print(f"      {window[:150]!r}")


if __name__ == "__main__":
    main()
