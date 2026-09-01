"""Compare what extraction finds across dataset versions and relaxation levels.

Two things change what comes out of a filing: how the text was produced, and how
tolerant the tokenizer was. This runs every combination of the two, on every
document, and reports where they disagree -- **across all eight citation kinds,
not just full case citations**, because a change that fixes a full citation can
silently strand the short forms that refer to it.

Three questions, in the order they are worth asking:

1. **Did anything change kind?** A census per citation type, per arm. A full
   citation becoming an unknown, or short forms appearing without their
   antecedent, shows here and nowhere else.
2. **Did an identifier change value?** The sharpest signal available. A case
   cited as `214 F.3d 1058` under one arm and `214 F.3d 1` under another is not
   a count difference -- both arms found one citation -- and no metric that
   counts citations can see it. Matching on volume and reporter and comparing
   the page is what catches it.
3. **Does anything look wrong on its face?** Heuristics over each citation's
   own text: a locator spanning a blank line, a reporter that kept whitespace,
   a page that looks like a margin line number. These are suspicions, not
   verdicts, and they are printed with their text so they can be read.

The default pair of levels is `bounded,full`, and that comparison is what the
tool exists for. `BOUNDED` is not a destination -- it is the instrument. It is
the widest relaxation with no known false positive, so anything `FULL` finds
that it does not is either a citation only `FULL` can reach or an error only
`FULL` makes, and there is nowhere else to see which. When `FULL` stops
producing disagreements worth keeping `BOUNDED` for, the level has done its job.

Datasets isolate the other factor. `v1` to `v1.1` is the Docling version alone;
`v1.1` to `v2.0` is the margin rule alone. Comparing `v1` to `v2.0` conflates
them, which is how 25 statute citations once looked like a margin-rule
regression in documents the margin rule had not touched.

    # every dataset on disk, at bounded and full
    uv run python scripts/compare_extractions.py

    # one dataset across all three relaxation levels
    uv run python scripts/compare_extractions.py --dataset v2=data/false-citation-bench-v2.0/documents_txt --levels all

    # everything, then read one document in full
    uv run python scripts/compare_extractions.py --levels all --document 022

Nothing here needs a model or a network, so it runs in seconds and is meant to
be re-run after every change.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from mellea_lrc.core.citations import (
    FullCaseCitation,
    FullJournalCitation,
    FullLawCitation,
    ShortCaseCitation,
)
from mellea_lrc.extraction import ExtractedCitation, Relaxation, extract_from_plain_text

BODY_MARKER = "--- Plain text ---\n"

# In the order that isolates one factor at a time: v1 to v1.1 is the Docling
# version alone, v1.1 to v2.0 is the margin rule alone. Comparing v1 to v2.0
# directly confounds the two, which is how 25 statute citations came to look
# like a margin-rule regression when the converter had changed underneath.
DEFAULT_DATASETS = {
    "v1": Path("data/false-citation-bench/documents_txt"),
    "v1.1": Path("data/false-citation-bench-v1.1/documents_txt"),
    "v2.0": Path("data/false-citation-bench-v2.0/documents_txt"),
}

# A page number this small, reached across a page break, is the pleading-paper
# failure: the margin line number stood in for the reporter page.
MARGIN_PAGE_CEILING = 40
_BLANK_LINE = re.compile(r"\n[ \t]*\n")


@dataclass(frozen=True, slots=True)
class Arm:
    """One (dataset, relaxation) pair, which is one way of reading the corpus."""

    dataset: str
    level: Relaxation

    def __str__(self) -> str:
        return f"{self.dataset}/{self.level.value}"


def body(path: Path) -> str:
    """The document body: the text after the provenance header, if there is one."""
    text = path.read_text(encoding="utf-8")
    _, marker, rest = text.partition(BODY_MARKER)
    return rest if marker else text


def kind(citation: ExtractedCitation) -> str:
    """The citation's type name, which is the dimension being compared."""
    return type(citation.citation).__name__


def locator(citation: ExtractedCitation) -> tuple[str, str, str] | None:
    """Volume, reporter and page, for the kinds that carry them."""
    inner = citation.citation
    if not isinstance(inner, FullCaseCitation | FullLawCitation | FullJournalCitation | ShortCaseCitation):
        return None
    volume, reporter, page = inner.volume, inner.reporter, inner.page
    if not (volume and reporter and page):
        return None
    return (str(volume).strip(), str(reporter).strip(), str(page).strip())


def identity(citation: ExtractedCitation) -> str:
    """What makes this citation the same citation across two renderings of a text.

    Offsets cannot be used: removing a margin moves every offset after it, so
    two versions of one document are different coordinate spaces. A locator
    identifies itself; anything else falls back to its text with whitespace
    flattened, which is what survives a re-render.
    """
    parts = locator(citation)
    if parts:
        return "|".join(parts)
    return " ".join(citation.matched_text.split())


def suspicions(citation: ExtractedCitation) -> list[str]:
    """Why this citation looks wrong on its own terms, if it does."""
    flags = []
    text = citation.matched_text
    parts = locator(citation)

    if _BLANK_LINE.search(text):
        flags.append("spans a blank line")
    if parts and parts[2].isdigit() and int(parts[2]) <= MARGIN_PAGE_CEILING and _BLANK_LINE.search(text):
        flags.append(f"page {parts[2]} reached across a break")
    if parts and parts[1] != parts[1].strip():
        flags.append("reporter kept whitespace")
    inner = citation.citation
    reporter = getattr(inner, "reporter", None)
    if reporter and reporter != reporter.strip():
        flags.append("reporter kept whitespace")
    if kind(citation) == "UnknownCitation":
        flags.append("unknown kind")
    if citation.resolves_to is None and kind(citation) in {
        "IdCitation",
        "SupraCitation",
        "ReferenceCitation",
    }:
        flags.append("back-reference with no antecedent")
    return flags


def run(paths: dict[str, Path], levels: list[Relaxation]) -> dict[Arm, dict[str, list[ExtractedCitation]]]:
    """Extract every document under every arm."""
    results: dict[Arm, dict[str, list[ExtractedCitation]]] = {}
    for dataset, directory in paths.items():
        documents = sorted(directory.glob("*.txt"))
        for level in levels:
            arm = Arm(dataset, level)
            per_document: dict[str, list[ExtractedCitation]] = {}
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                for path in documents:
                    extracted = extract_from_plain_text(body(path), relaxation=level)
                    per_document[path.stem] = list(extracted.citations)
            results[arm] = per_document
    return results


def census(results: dict[Arm, dict[str, list[ExtractedCitation]]]) -> None:
    """Counts by citation kind, per arm. Every kind, not just full citations."""
    kinds: set[str] = set()
    counts: dict[Arm, Counter] = {}
    for arm, documents in results.items():
        tally: Counter = Counter()
        for citations in documents.values():
            tally.update(kind(c) for c in citations)
        counts[arm] = tally
        kinds |= set(tally)

    ordered = sorted(kinds)
    width = max(len(str(arm)) for arm in results) + 2
    print(f"{'arm':<{width}}" + "".join(f"{k[:15]:>17}" for k in ordered) + f"{'TOTAL':>9}")
    for arm, tally in counts.items():
        row = "".join(f"{tally.get(k, 0):>17}" for k in ordered)
        print(f"{arm!s:<{width}}{row}{sum(tally.values()):>9}")


def value_changes(results: dict[Arm, dict[str, list[ExtractedCitation]]], baseline: Arm) -> None:
    """Citations occupying the same place in the text that report a different identifier.

    This is the failure a count cannot see: one citation before, one after, a
    different case named. `214 F.3d 1058` becoming `214 F.3d 1` is the case.

    Two citations are the same citation when their locator spans overlap, which
    is only meaningful **within one dataset**, where every arm reads the same
    characters. Across datasets the offsets are different coordinate spaces --
    removing a margin moves everything after it -- so those pairs are left to
    `membership_changes`, which compares sets rather than positions.

    Matching on volume and reporter instead would be wrong in the ordinary
    case: a filing citing two sections of one statute, or two cases from one
    reporter volume, is not a document that changed.
    """
    print("\n=== same position, different identifier (within a dataset) ===")
    found = False
    for dataset in dict.fromkeys(arm.dataset for arm in results):
        arms = [arm for arm in results if arm.dataset == dataset]
        base = baseline if baseline in arms else arms[0]
        for arm in arms:
            if arm == base:
                continue
            for document, citations in results[arm].items():
                before = [c for c in results[base].get(document, []) if locator(c)]
                for citation in citations:
                    if not (parts := locator(citation)):
                        continue
                    for other in before:
                        overlaps = (
                            citation.locator_span.start < other.locator_span.end
                            and other.locator_span.start < citation.locator_span.end
                        )
                        if overlaps and locator(other) != parts:
                            found = True
                            was = " ".join(locator(other) or ())
                            now = " ".join(parts)
                            print(f"  {base} -> {arm}  {document[:30]}  {was}  ->  {now}")
    if not found:
        print("  (none)")


def membership_changes(
    results: dict[Arm, dict[str, list[ExtractedCitation]]], baseline: Arm, limit: int
) -> None:
    """What each arm gains and loses against the baseline, by kind."""
    print(f"\n=== gained and lost against {baseline}, by kind ===")
    base = results[baseline]
    for arm, documents in results.items():
        if arm == baseline:
            continue
        gained: dict[str, list[str]] = defaultdict(list)
        lost: dict[str, list[str]] = defaultdict(list)
        for document, citations in documents.items():
            before = {identity(c): c for c in base.get(document, [])}
            after = {identity(c): c for c in citations}
            for key in after.keys() - before.keys():
                gained[kind(after[key])].append(f"{document[:22]}: {key[:52]}")
            for key in before.keys() - after.keys():
                lost[kind(before[key])].append(f"{document[:22]}: {key[:52]}")
        print(f"\n  --- {arm} ---")
        for label, table in (("gained", gained), ("lost", lost)):
            if not table:
                print(f"    {label}: (none)")
                continue
            for citation_kind, entries in sorted(table.items()):
                print(f"    {label} {citation_kind} x{len(entries)}")
                for entry in entries[:limit]:
                    print(f"       {entry}")


def suspicious(results: dict[Arm, dict[str, list[ExtractedCitation]]], limit: int) -> None:
    """Everything that looks wrong on its face, per arm, with its text."""
    print("\n=== suspicious positives ===")
    for arm, documents in results.items():
        rows = [
            (document, flag, " ".join(c.matched_text.split())[:64])
            for document, citations in documents.items()
            for c in citations
            for flag in suspicions(c)
        ]
        tally = Counter(flag for _, flag, _ in rows)
        print(f"\n  --- {arm} --- {dict(tally) if tally else '(clean)'}")
        interesting = [r for r in rows if r[1] != "back-reference with no antecedent"]
        for document, flag, text in interesting[:limit]:
            print(f"    {document[:26]:<28}{flag:<34}{text!r}")


def drill(results: dict[Arm, dict[str, list[ExtractedCitation]]], needle: str) -> None:
    """Every citation of one document, under every arm, side by side."""
    print(f"\n=== document matching {needle!r} ===")
    for arm, documents in results.items():
        for document, citations in documents.items():
            if needle not in document:
                continue
            print(f"\n  --- {arm}  {document[:56]} ({len(citations)} citations) ---")
            for citation in citations:
                flags = suspicions(citation)
                mark = "  <-- " + ", ".join(flags) if flags else ""
                text = " ".join(citation.matched_text.split())[:56]
                print(f"    {kind(citation):<20}{text!r:<60}{mark}")


def main() -> int:
    """Run every arm and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        metavar="LABEL=PATH",
        help="a labelled directory of .txt documents; repeatable. Defaults to every known version on disk.",
    )
    parser.add_argument(
        "--levels",
        default="bounded,full",
        help=(
            "'all', or a comma-separated subset of none,bounded,full "
            "(default: bounded,full -- the pair whose disagreement is the point)"
        ),
    )
    parser.add_argument("--baseline", help="arm to compare against, as dataset/level (default: the first)")
    parser.add_argument("--document", help="print every citation of the documents matching this substring")
    parser.add_argument("--limit", type=int, default=8, help="rows to show per section (default: 8)")
    args = parser.parse_args()

    if args.dataset:
        paths = {}
        for entry in args.dataset:
            label, _, path = entry.partition("=")
            paths[label] = Path(path)
    else:
        paths = {label: path for label, path in DEFAULT_DATASETS.items() if path.is_dir()}

    missing = [f"{label}={path}" for label, path in paths.items() if not path.is_dir()]
    if missing or not paths:
        print(f"no readable datasets: {missing or 'none given and none found on disk'}")
        return 1

    levels = (
        list(Relaxation)
        if args.levels == "all"
        else [Relaxation(name.strip()) for name in args.levels.split(",")]
    )

    print("datasets: " + ", ".join(f"{k} ({len(list(v.glob('*.txt')))} docs)" for k, v in paths.items()))
    print("levels:   " + ", ".join(level.value for level in levels) + "\n")

    results = run(paths, levels)
    census(results)

    baseline = next(iter(results))
    if args.baseline:
        dataset, _, level = args.baseline.partition("/")
        baseline = Arm(dataset, Relaxation(level))

    value_changes(results, baseline)
    membership_changes(results, baseline, args.limit)
    suspicious(results, args.limit)
    if args.document:
        drill(results, args.document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
