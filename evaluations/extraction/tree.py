"""Score an extraction run against the citation tree ground truth.

The bench `evaluate.py` reads is a flat list of identifiers: it asks whether a
citation was found and nothing else. This one reads `extraction-v3.0`, which is
a tree -- every place a filing cites a case, which place introduced the case,
and which page each one claims -- and asks the three questions separately,
because a pass that finds every citation and files half of them under the wrong
case is not doing better than one that finds fewer.

## The report

Every metric is `<thing>_parsed` or `<thing>_attributed`, and every one of them
is counted **against the ground truth's own denominator**: `pincite_parsed` is
out of the pin cites the filings state, not out of the citations this run
happened to find. A run that reads fewer citations therefore reads fewer pin
cites, which is the point -- a denominator that shrinks with the run would hide
it.

Detail lines under a metric are indented with `- `. Nothing else is printed.

## Matching

An annotated citation is **parsed** when the run produces a citation at exactly
its `locator` span, or at its `cited_as` span where it has no locator, which is
`Id.` and the bare-name references. Nothing partial counts: `PROTOCOL.md` fixes
that rule, and the identifier is what a lookup resolves.

A root is named by its span rather than by an id, so a run that knows nothing
about this dataset's ids can still be scored: a citation is **attributed** when
the run's root sits at the span of the row the ground truth calls its root.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mellea_lrc.core.citations import CitationKind, citation_kind
from mellea_lrc.extraction import Relaxation, extract_from_plain_text

# Every kind that cites a case. A statute is not one, and the tree says nothing
# about it, so reporting one is neither right nor wrong here.
CASE_KINDS = frozenset(
    {
        CitationKind.FULL_CASE,
        CitationKind.DOCKET,
        CitationKind.SHORT_CASE,
        CitationKind.ID,
        CitationKind.REFERENCE,
        CitationKind.SUPRA,
    }
)


def anchor(row: dict[str, Any]) -> tuple[int, int]:
    """The span an annotated citation is matched at.

    The locator, or the whole citation where the row states no locator: `Id.`
    and a bare name identify nothing, which is what makes them short forms.
    """
    span = row.get("locator") or row["cited_as"]
    return (span["start"], span["end"])


def read_documents(dataset: Path) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Each document's header row and its annotations."""
    for path in sorted(dataset.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        yield rows[0], rows[1:]


def run_document(text: str, relaxation: Relaxation) -> list[dict[str, Any]]:
    """Run extraction over one document and project the tree artifact rows."""
    # Eyecite writes overlap diagnostics to stdout as it reads.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        extracted = extract_from_plain_text(text, relaxation=relaxation)
    at = {citation.citation_id: citation.locator_span for citation in extracted.citations}
    rows = []
    for citation in extracted.citations:
        if citation_kind(citation.citation) not in CASE_KINDS:
            continue
        root = at.get(citation.root_id or "")
        rows.append(
            {
                "kind": citation_kind(citation.citation).value,
                "span": {"start": citation.locator_span.start, "end": citation.locator_span.end},
                "root": {"start": root.start, "end": root.end} if root else None,
                "pin_cite": (
                    {
                        "start": citation.pin_cite_span.start,
                        "end": citation.pin_cite_span.end,
                        "pages": [
                            {"first": page.first, "last": page.last, "kind": page.kind.value}
                            for page in citation.pin_cite_pages
                        ],
                    }
                    if citation.pin_cite_span
                    else None
                ),
            }
        )
    return rows


def score(dataset: Path, corpus: Path, relaxation: Relaxation) -> tuple[Counter[str], dict[str, list[str]]]:
    """Score one relaxation over the whole corpus."""
    counts: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    detail: dict[str, list[str]] = {"citation_parsed": [], "pincite_parsed": [], "root_attributed": []}
    for header, body in read_documents(dataset):
        text = (corpus / header["document"]).read_text(encoding="utf-8")
        run = {(row["span"]["start"], row["span"]["end"]): row for row in run_document(text, relaxation)}
        annotated = [row for row in body if row["unit"] == "citation"]
        parsed = {row["id"]: run[anchor(row)] for row in annotated if anchor(row) in run}
        for row in annotated:
            counts["citation"] += 1
            by_kind[f"{row['kind']}:of"] += 1
            found = parsed.get(row["id"])
            if found is None:
                detail["citation_parsed"].append(
                    f"{row['id']} {row['kind']} {row['cited_as']['quote'][:48]!r}"
                )
            else:
                counts["citation_parsed"] += 1
                by_kind[f"{row['kind']}:parsed"] += 1

            counts["root"] += 1
            expected = parsed.get(row["root_id"])
            if found is not None and expected is not None and found["root"] == expected["span"]:
                counts["root_attributed"] += 1
            elif found is not None:
                detail["root_attributed"].append(f"{row['id']} root {row['root_id']} not reached")

            want = row.get("pin_cite")
            if want is None:
                if found is not None and found["pin_cite"] is not None:
                    detail["pincite_parsed"].append(f"{row['id']} reads a pin cite the filing does not state")
                continue
            counts["pincite"] += 1
            got = found["pin_cite"] if found else None
            if got is None:
                detail["pincite_parsed"].append(f"{row['id']} {want['quote']!r} not read")
            elif (got["start"], got["end"]) != (want["start"], want["end"]):
                detail["pincite_parsed"].append(f"{row['id']} {want['quote']!r} read at another span")
            elif got["pages"] != want["normalized"]:
                detail["pincite_parsed"].append(f"{row['id']} {want['quote']!r} reads as {got['pages']}")
            else:
                counts["pincite_parsed"] += 1
    counts.update(by_kind)
    return counts, detail


def report(counts: Counter[str], detail: dict[str, list[str]], kinds: list[str]) -> str:
    """Render one run: a line per metric, and its disagreements under it."""
    lines = []
    for metric, over in (
        ("citation_parsed", "citation"),
        ("pincite_parsed", "pincite"),
        ("root_attributed", "root"),
    ):
        lines.append(f"{metric:<18}{counts[metric]}/{counts[over]}")
        if metric == "citation_parsed":
            lines += [f"- {kind:<20}{counts[f'{kind}:parsed']}/{counts[f'{kind}:of']}" for kind in kinds]
        lines += [f"- {line}" for line in detail[metric]]
    return "\n".join(lines)


def main() -> None:
    """Run the command-line tree evaluator."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True, help="extraction-v3.0/documents/")
    parser.add_argument("--documents", type=Path, required=True, help="The text those spans index.")
    parser.add_argument(
        "--relaxation",
        default="FULL",
        choices=[level.name for level in Relaxation],
        help="Which tokenizer to read with.",
    )
    args = parser.parse_args()

    relaxation = Relaxation[args.relaxation]
    counts, detail = score(args.dataset, args.documents, relaxation)
    kinds = sorted({key.split(":")[0] for key in counts if key.endswith(":of")})
    print(f"== {relaxation.name} ==")
    print(report(counts, detail, kinds))


if __name__ == "__main__":
    main()
