"""Do either of the hunters reach the citations the extractor misses?

The bench records which locators are stated and the site scorer says which are
missed. This asks the next question: of the ones missed, how many does the
residue pipeline even put in front of a model? A citation neither hunter
reaches cannot be recovered by adjudication at any price, because nothing ever
asks about it.

    uv run python -m exploration.locator_recall.check_reach
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import io
import json
from pathlib import Path

from exploration.locator_recall.fuzzy_sites import SITE, body, gazetteer, letters_only
from exploration.locator_recall.residue import FUZZY_MIN_LETTERS
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.adjudication import mask_full_spans, suspected_locators


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Whether two half-open spans share a character."""
    return a[0] < b[1] and b[0] < a[1]


def net_spans(masked: str, known: dict[str, str], keys: list[str], threshold: float) -> list[tuple[int, int]]:
    """Where the fuzzy net would report a candidate in this masked text."""
    spans = []
    for match in SITE.finditer(masked):
        key = letters_only(match.group(3))
        if not key:
            continue
        if len(key) < FUZZY_MIN_LETTERS:
            close = [key] if key in known else []
        else:
            close = difflib.get_close_matches(key, keys, n=1, cutoff=threshold)
        if close:
            spans.append((match.start(), match.end()))
    return spans


def main() -> int:
    """For every ground-truth locator the extractor misses, say who reaches it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=Path("data/false-citation-bench-locator-only-v2.0"))
    parser.add_argument("--threshold", type=float, default=0.67)
    args = parser.parse_args()

    known = gazetteer()
    keys = list(known)
    records = [
        json.loads(line)
        for line in (args.bench / "extraction.jsonl").read_text().splitlines()
        if line.strip()
    ]

    print(f"{'document':<26}{'expected':<24}{'hunting':<10}{'fuzzy':<8}")
    missed_total = reached = 0

    for path in sorted((args.bench / "documents_txt").glob("*.txt")):
        mine = [r for r in records if r["document"] == path.name]
        if not mine:
            continue
        text = body(path)

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
            hunted = suspected_locators(document)
        masked = mask_full_spans(document)

        extracted = [(c.locator_span.start, c.locator_span.end) for c in document.citations]
        hunted_spans = [(s.span_start, s.span_end) for s in hunted]
        fuzzy = net_spans(masked, known, keys, args.threshold)

        for record in mine:
            span = (record["span"]["start"], record["span"]["end"])
            if any(overlaps(span, other) for other in extracted):
                continue
            missed_total += 1
            in_hunting = any(overlaps(span, other) for other in hunted_spans)
            in_net = any(overlaps(span, other) for other in fuzzy)
            reached += 1 if (in_hunting or in_net) else 0
            expected = f"{record['volume']} {record['reporter']} {record['page']}"
            print(
                f"{path.stem[:24]:<26}{expected[:22]:<24}"
                f"{('yes' if in_hunting else 'NO'):<10}{('yes' if in_net else 'NO'):<8}"
            )

    print(f"\nlocators the extractor misses: {missed_total}")
    print(f"  reached by at least one hunter: {reached}")
    print(f"  reached by neither:             {missed_total - reached}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
