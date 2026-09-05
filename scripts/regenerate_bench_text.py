"""Re-export the benchmark's plain text with the margin rule on.

`data/false-citation-bench/documents_txt/` was exported by a Docling run that
predates `reclassify_margin_line_numbers`. The rule works on the Docling
document during conversion and cannot reach an already-exported `.txt`, so the
published text still carries every pleading-paper margin.

This reconverts the published PDFs with the rule on and writes the result
beside the original as a new dataset version. It also reports, per document,
how many margin numbers were removed and whether a gutter survived the rule --
which is the open question in `exploration/notes/pleading-paper-margins.md`.

**The new text is a different coordinate space.** Removing a margin moves the
offset of everything after it, so the published annotations, whose spans are
anchored to the v1 text, do not transfer. That is why this writes a new version
rather than overwriting, and why the output carries a README saying so.

    uv run python scripts/regenerate_bench_text.py

Docling reconversion is slow -- minutes for the corpus, not seconds.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from mellea_lrc.preprocessing import preprocess_with_docling

BODY_MARKER = "--- Plain text ---\n"

# A surviving gutter, read off the text rather than the layout: four or more
# consecutive ascending integers, each alone on its own line. Four is enough to
# separate a margin from a numbered list that happens to start at 1, and the
# rule's own threshold is higher than that, so anything this finds is something
# the rule declined to take.
_STANDALONE_NUMBERS = re.compile(r"(?m)^[ \t]*(\d{1,3})[ \t]*$")
_MIN_RUN = 4


def body_of(path: Path) -> str:
    """The document body: the text after the provenance header."""
    text = path.read_text(encoding="utf-8")
    _, marker, body = text.partition(BODY_MARKER)
    return body if marker else text


def gutter_runs(text: str) -> list[list[int]]:
    """Runs of consecutive ascending integers left standing alone in the text."""
    numbers = [int(match.group(1)) for match in _STANDALONE_NUMBERS.finditer(text)]
    runs: list[list[int]] = []
    current: list[int] = []
    for value in numbers:
        if current and value == current[-1] + 1:
            current.append(value)
            continue
        if len(current) >= _MIN_RUN:
            runs.append(current)
        current = [value]
    if len(current) >= _MIN_RUN:
        runs.append(current)
    return runs


def header(source: Path, backend_version: str | None) -> str:
    """The provenance header the published corpus writes above its body."""
    return f"Source PDF: {source}\nBackend: docling\nBackend version: {backend_version}\n\n{BODY_MARKER}"


MARGIN_README = """# false-citation-bench, margin-adjusted (v2.0)

`documents_txt/` re-exported from `../false-citation-bench/documents_pdf/` with
`mellea_lrc.preprocessing.margin_line_numbers` on.

**Compare it against v1.1, not against v1.** The published v1 text was produced
by an older Docling, so a v1-to-v2.0 difference confounds the margin rule with
ten minor versions of the converter -- which showed up as 25 statute citations
disappearing in documents the margin rule never touched. v1.1 is this same
converter with the rule off, and is the only control that isolates it.

**The annotations do not transfer.** Removing a pleading-paper margin moves the
offset of everything after it, so every span in
`../false-citation-bench/annotations/` and `../false-citation-bench/derived/`
is anchored to v1 text and is wrong against this. Re-anchoring them is a
separate piece of work and has not been done. Nothing here is published.

Produced by `scripts/regenerate_bench_text.py` on branch
`preprocess/margin-line-numbers`. See `gutter-report.md` beside this file for
what the rule removed and what it left.
"""

CONTROL_README = """# false-citation-bench, reconverted (v1.1)

`documents_txt/` re-exported from `../false-citation-bench/documents_pdf/` with
the margin rule **off**, so its only difference from the published v1 is the
Docling version that produced it.

This exists to be the control for v2.0. v1 was converted with Docling 2.105.0
and everything since with a later one, so a v1-to-v2.0 comparison cannot say
whether a change came from the margin rule or from the converter. v1.1 splits
that: v1 to v1.1 is the converter alone, v1.1 to v2.0 is the margin rule alone.

The annotations do not transfer to this either -- a converter changes offsets
as readily as a margin rule does. Nothing here is published.

Produced by `scripts/regenerate_bench_text.py --no-margin` on branch
`preprocess/margin-line-numbers`.
"""


def main() -> int:
    """Reconvert the corpus and report on the margins."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdfs", type=Path, default=Path("data/false-citation-bench/documents_pdf"))
    parser.add_argument("--v1", type=Path, default=Path("data/false-citation-bench/documents_txt"))
    parser.add_argument("--out", type=Path, help="default depends on --no-margin")
    parser.add_argument(
        "--no-margin",
        action="store_true",
        help="leave the margin in, producing the v1.1 control: same converter, rule off",
    )
    args = parser.parse_args()

    drop_margins = not args.no_margin
    if args.out is None:
        args.out = Path(
            "data/renderings/false-citation-bench-v2.0"
            if drop_margins
            else "data/renderings/false-citation-bench-v1.1"
        )

    sources = sorted(args.pdfs.glob("*.pdf"))
    if not sources:
        print(f"{args.pdfs}: no PDFs found", file=sys.stderr)
        return 1

    text_dir = args.out / "documents_txt"
    text_dir.mkdir(parents=True, exist_ok=True)
    (args.out / "README.md").write_text(MARGIN_README if drop_margins else CONTROL_README, encoding="utf-8")

    rows = []
    for index, source in enumerate(sources, start=1):
        print(f"[{index}/{len(sources)}] {source.name}", file=sys.stderr, flush=True)
        document = preprocess_with_docling(source, drop_margin_line_numbers=drop_margins)

        destination = text_dir / f"{source.stem}.txt"
        destination.write_text(
            header(source, document.preprocessing_metadata.backend_version) + document.text,
            encoding="utf-8",
        )

        before = args.v1 / f"{source.stem}.txt"
        rows.append(
            {
                "document": source.stem,
                "dropped": document.preprocessing_metadata.margin_line_numbers_dropped,
                "before": gutter_runs(body_of(before)) if before.exists() else None,
                "after": gutter_runs(document.text),
            }
        )

    write_report(args.out / "gutter-report.md", rows)
    return 0


def write_report(path: Path, rows: list[dict]) -> None:
    """Write, and print, what the rule removed and what survived it."""
    had = [r for r in rows if r["before"]]
    left = [r for r in rows if r["after"]]
    total = sum(r["dropped"] or 0 for r in rows)

    lines = [
        "# What the margin rule removed, and what it left",
        "",
        f"- documents reconverted: **{len(rows)}**",
        f"- margin line numbers removed: **{total:,}**",
        f"- documents carrying a gutter before: **{len(had)}**",
        f"- documents carrying a gutter after: **{len(left)}**",
        "",
        "A gutter here is four or more consecutive ascending integers each",
        "standing alone on its own line, read off the exported text.",
        "",
        "| document | removed | gutter runs before | gutter runs after |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        before = "n/a" if row["before"] is None else str(len(row["before"]))
        lines.append(f"| `{row['document'][:44]}` | {row['dropped']} | {before} | {len(row['after'])} |")

    if left:
        lines += ["", "## Documents still carrying a gutter", ""]
        for row in left:
            runs = "; ".join(f"{run[0]}–{run[-1]} ({len(run)})" for run in row["after"][:6])
            lines.append(f"- `{row['document']}` — {runs}")

    report = "\n".join(lines) + "\n"
    path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    raise SystemExit(main())
