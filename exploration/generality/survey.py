r"""Do the extraction fixes hold on documents they were not built from?

`false-citation-bench` is 26 documents and every rule in this project was read
off it. `local/mined-corpus` is 77 further filings, mined separately and never
looked at while any of this was written, which makes it the only evidence that
the rules generalise rather than fit.

This runs the same measurements over both and prints them side by side.

    uv run python -m exploration.generality.survey
    uv run python -m exploration.generality.survey --corpus <dir>
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
from collections import Counter
from pathlib import Path

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.structure.colocation import colocation_groups

BODY_MARKER = "--- Plain text ---\n"
BENCH = Path("data/extraction-v2.0/documents_txt")
MINED = Path.home() / "CodingProjects/mellea-lrc/local/mined-corpus"

# A pin cite that landed in `extra`: digits, star pages, paragraph marks. Not a
# section, which names a statute, and not a parallel citation.
PIN_SHAPED = re.compile(r"^[*¶]?\s*\d[\d\s,\-–*¶n.]*$")
PARALLEL = re.compile(r"\d+\s+[A-Z][A-Za-z.'’ ]*\d*\s+\d+")


def body(path: Path) -> str:
    """The document text, past the header some corpora carry."""
    text = path.read_text(encoding="utf-8")
    _, marker, rest = text.partition(BODY_MARKER)
    return rest if marker else text


def survey(directory: Path) -> tuple[Counter, list[tuple[str, str, str]]]:
    """Measure one corpus, and collect the citations whose metadata crossed."""
    counts: Counter = Counter()
    crossings: list[tuple[str, str, str]] = []
    written: set[str] = set()
    canonical: set[str] = set()

    for path in sorted(directory.glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
        counts["documents"] += 1
        counts["colocation groups"] += len(colocation_groups(document.citations))
        citations = list(document.citations)

        for item in citations:
            citation = item.citation
            reporter = getattr(citation, "reporter", None)
            if reporter is not None:
                written.add(reporter.as_written)
                canonical.add(reporter.canonical)
            if not isinstance(citation, FullCaseCitation):
                continue
            counts["case citations"] += 1
            counts["with a date"] += bool(citation.date)
            counts["with an exact date"] += bool(citation.date and citation.date.is_exact)
            counts["with a court"] += bool(citation.court)
            counts["with a pin cite"] += bool(citation.pin_cite)
            counts["with both parties"] += bool(citation.plaintiff and citation.defendant)
            counts["with no party at all"] += not (citation.plaintiff or citation.defendant)

            extra = (citation.extra or "").strip()
            if extra and not PARALLEL.search(extra) and PIN_SHAPED.match(extra):
                counts["pin cite lost to extra"] += 1

            # A span still covering an unrelated citation means metadata crossed.
            forward = [
                other
                for other in citations
                if other is not item
                and item.locator_span.end <= other.locator_span.start
                and other.locator_span.end <= item.full_span.end
                and not (item.colocation_id and other.colocation_id == item.colocation_id)
            ]
            backward = [
                other
                for other in citations
                if other is not item
                and other.locator_span.end <= item.locator_span.start
                and other.locator_span.start >= item.full_span.start
                and not (item.colocation_id and other.colocation_id == item.colocation_id)
            ]
            if forward:
                counts["span still crosses forward"] += 1
                crossings.append((path.stem[:14], "forward", item.matched_text[:20]))
            if backward:
                counts["span crosses backward"] += 1
                crossings.append((path.stem[:14], "backward", item.matched_text[:20]))

    counts["reporter spellings as written"] = len(written)
    counts["reporter spellings canonical"] = len(canonical)
    return counts, crossings


def main() -> int:
    """Print both corpora side by side, as counts and as rates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=BENCH)
    parser.add_argument("--mined", type=Path, default=MINED)
    args = parser.parse_args()

    bench, _ = survey(args.bench)
    mined, mined_crossings = survey(args.mined)

    labels = [
        "documents",
        "case citations",
        "with a date",
        "with an exact date",
        "with a court",
        "with a pin cite",
        "with both parties",
        "with no party at all",
        "pin cite lost to extra",
        "span still crosses forward",
        "span crosses backward",
        "colocation groups",
        "reporter spellings as written",
        "reporter spellings canonical",
    ]
    print(f"{'':<32}{'bench (tuned on)':>20}{'mined (unseen)':>20}")
    for label in labels:
        left, right = bench[label], mined[label]
        rate = ""
        if label not in {"documents", "colocation groups"} and not label.startswith("reporter"):
            lp = 100 * left / bench["case citations"] if bench["case citations"] else 0
            rp = 100 * right / mined["case citations"] if mined["case citations"] else 0
            rate = f"   {lp:>5.1f}% vs {rp:>5.1f}%"
        print(f"{label:<32}{left:>20}{right:>20}{rate}")

    print("\ncitations whose span still crosses another, on the unseen corpus:")
    for row in mined_crossings[:20]:
        print(f"  {row[0]:<16}{row[1]:<10}{row[2]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
