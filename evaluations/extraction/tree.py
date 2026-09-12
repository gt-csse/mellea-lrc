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

The last two metrics are the other side of the corpus. `noncase_parsed` is how
many of the 27 `noncase_citation` rows the run reports: the filing does write
`Id.` and a shortened cite at those places, so reading them is right, and what
the ground truth says is only that they point at something other than a
decision. `unaccounted` is the run's case citations that no row accounts for at
all, out of every case-kind citation it reported -- the precision side, and the
one number here where lower is better.

Statutes and journal citations are not in this ground truth and are not counted
either way. A statute *read as a case* is a case-kind citation at a span no row
claims, so it lands in `unaccounted`, which is where it belongs.

An unaccounted citation is not always an invented one. A citation read at a span
no row has is unaccounted too: unrelaxed eyecite reports `673 F.2d at ` where
the filing writes `673 F.2d at 57`, and that is one miss and one unaccounted
report of the same citation. The detail line carries the text, which says which
it is.

## Matching

`root_parsed` is the roots alone -- the citations that state an identifier for
the first time, 427 of the 703. A root reached is a case the filing can be
checked for; a short form missed costs a page claim, and a root missed costs the
case.

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
    detail: dict[str, list[str]] = {
        "citation_parsed": [],
        "root_parsed": [],
        "pincite_parsed": [],
        "root_attributed": [],
        "noncase_parsed": [],
        "unaccounted": [],
    }
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
            if row["is_root"]:
                counts["root"] += 1
                if found is None:
                    detail["root_parsed"].append(
                        f"{row['id']} {row['kind']} {row['cited_as']['quote'][:48]!r}"
                    )
                else:
                    counts["root_parsed"] += 1

            counts["citation_with_root"] += 1
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

        # The other side of the same corpus: what the run reports that no
        # citation row claims. A noncase row accounts for its own span -- the
        # filing does write `Id.` there and a reader is right to read it -- so
        # reporting one is neither a hit nor an error, and is counted apart.
        claimed = {(row["span"]["start"], row["span"]["end"]) for row in parsed.values()}
        noncase = [
            (row["cited_as"]["start"], row["cited_as"]["end"], row["id"])
            for row in body
            if row["unit"] == "noncase_citation"
        ]
        counts["noncase"] += len(noncase)
        reported = set()
        for span in run:
            counts["case_kind_reported"] += 1
            if span in claimed:
                continue
            inside = next((r for start, end, r in noncase if start <= span[0] and span[1] <= end), None)
            if inside is not None:
                reported.add(inside)
                continue
            counts["unaccounted"] += 1
            detail["unaccounted"].append(
                f"{header['document'][:3]} {run[span]['kind']} {text[span[0] : span[1]][:48]!r}"
            )
        counts["noncase_parsed"] += len(reported)
        detail["noncase_parsed"] += [f"{row} not read" for _, _, row in noncase if row not in reported]
    counts.update(by_kind)
    return counts, detail


def report(counts: Counter[str], detail: dict[str, list[str]], kinds: list[str]) -> str:
    """Render one run: a line per metric, and its disagreements under it."""
    lines = []
    for metric, over in (
        ("citation_parsed", "citation"),
        ("root_parsed", "root"),
        ("pincite_parsed", "pincite"),
        ("root_attributed", "citation_with_root"),
        ("noncase_parsed", "noncase"),
        ("unaccounted", "case_kind_reported"),
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
