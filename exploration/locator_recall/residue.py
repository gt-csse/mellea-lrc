"""What is left in a filing once every citation the extractor found is masked out.

The order matters, and getting it wrong is what made an earlier version of this
report hundreds of court parentheticals as candidates.

    1. extract with the widest deterministic setting there is -- margin-adjusted
       text, FULL relaxation -- so the extractor is given every chance first
    2. mask every citation it produced, full span, which is primary and
       secondary alike: a short form, an `id.`, a `supra` and the case name and
       parenthetical around each of them
    3. hunt only what is left

Masking full spans rather than locators is what removes the noise. `(D. Ariz.
2017)` is a court parenthetical, and `Ariz.` is a reporter string in eyecite's
own gazetteer, so on unmasked text every citation in the corpus generates a
spurious site immediately after itself.

Two hunters run over the residue and they fail differently:

    site hunting   an exact gazetteer reporter with digits close on both sides.
                   Cannot see a reporter the converter misspelled.
    the fuzzy net  a number, some letters, a number, with the letters reduced to
                   letters alone and matched by similarity. Sees a misspelled
                   reporter; pays for it in noise.

Their union is the candidate set, and the point of masking first is that the set
is small enough to put a model on. Adjudication is a separate step and is not
run here -- this says how much there is to adjudicate, and what it looks like.

    uv run python -m exploration.locator_recall.residue
    uv run python -m exploration.locator_recall.residue --show 40
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import io
from collections import Counter
from pathlib import Path

from exploration.locator_recall.fuzzy_sites import SITE, body, gazetteer, letters_only, shape
from mellea_lrc.experimental import mask_full_spans, suspected_locators
from mellea_lrc.extraction import Relaxation, extract_from_plain_text

FUZZY_MIN_LETTERS = 4


def main() -> int:
    """Mask what was found, hunt what is left, and report both hunters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=Path("data/false-citation-bench-locator-only-v2.0"))
    parser.add_argument("--threshold", type=float, default=0.67)
    parser.add_argument("--show", type=int, default=20)
    args = parser.parse_args()

    known = gazetteer()
    keys = list(known)

    totals: Counter = Counter()
    shapes: Counter = Counter()
    candidates: list[tuple[str, str, str, str]] = []

    for path in sorted((args.bench / "documents_txt").glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
            hunted = suspected_locators(document)
        masked = mask_full_spans(document)

        totals["citations found"] += len(document.citations)
        totals["characters masked"] += sum(1 for a, b in zip(text, masked, strict=True) if a != b)

        hunted_spans = set()
        for site in hunted:
            totals["site hunting"] += 1
            hunted_spans.add((site.span_start, site.span_end))
            candidates.append((path.stem[:20], "hunting", site.reporter, site.window[:120]))

        # The net reads the masked text, so a citation already found cannot
        # produce a candidate and neither can the court parenthetical beside it.
        for match in SITE.finditer(masked):
            key = letters_only(match.group(3))
            if not key:
                continue
            if len(key) < FUZZY_MIN_LETTERS:
                close = [key] if key in known else []
            else:
                close = difflib.get_close_matches(key, keys, n=1, cutoff=args.threshold)
            if not close:
                continue
            totals["fuzzy net"] += 1
            start, end = match.start(), match.end()
            if any(start < b and a < end for a, b in hunted_spans):
                totals["both"] += 1
                continue
            totals["fuzzy net only"] += 1
            label = shape(masked, start, end, match)
            shapes[label] += 1
            window = " ".join(text[max(0, start - 70) : end + 70].split())
            candidates.append((path.stem[:20], f"fuzzy/{label}", known[close[0]], window[:120]))

    print("| stage | count |")
    print("|---|---:|")
    for label in (
        "citations found",
        "characters masked",
        "site hunting",
        "fuzzy net",
        "both",
        "fuzzy net only",
    ):
        print(f"| {label} | {totals[label]} |")

    print("\n| what the fuzzy-only candidates are | count |")
    print("|---|---:|")
    for label, count in shapes.most_common():
        print(f"| {label} | {count} |")

    to_adjudicate = totals["site hunting"] + totals["fuzzy net only"]
    print(f"\ncandidates a model would be asked about: **{to_adjudicate}**")

    print(f"\n--- showing {args.show} ---")
    for stem, method, reporter, window in candidates[: args.show]:
        print(f"  [{stem:<20}] {method:<28} ~ {reporter!r}")
        print(f"      {window!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
