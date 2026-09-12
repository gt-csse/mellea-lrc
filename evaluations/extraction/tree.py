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

`citation_misparsed` is the other side, and the one number here where lower is
better: everything the run reads as a citation to a case that is **not** one of
the 703, out of every case-kind citation it reported. Two things land there.

Twenty-seven places are written exactly like a short form and point at something
that is not a decision -- a pleading's numbered allegations, an exhibit
declaration, a statute, a case quoted inside another case. `Rosenblatt v. Baer,
383 U.S. at 85` parses as a short case citation, and it is Anaya's citation
rather than this filing's, so the page it claims cannot be checked against an
opinion the filing relies on. The ground truth holds those as
`noncase_citation` rows, so the detail line says which one was hit.

The rest are citations at a span no row has. That includes a citation read with
the wrong edges: unrelaxed eyecite reports `673 F.2d at ` where the filing
writes `673 F.2d at 57`, which is one miss and one misparse of the same
citation.

Statutes and journal citations are not in this ground truth and are not counted
either way -- but a statute *read as a case* is a case-kind citation at a span
no row claims, so it lands here, which is where it belongs.

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
                "is_root": root is not None
                and (root.start, root.end) == (citation.locator_span.start, citation.locator_span.end),
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
    """Score one relaxation over the whole corpus.

    Each measure is counted twice. **Recall** is out of what the filings state,
    so it says what a run misses. **Precision** is out of what the run reports,
    so it says what a run makes up. One without the other hides half of a pass:
    a reader that reports every span in the document has perfect recall.
    """
    counts: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    detail: dict[str, list[str]] = {
        key: [] for key in ("citation", "root", "short_form", "pincite", "attribution")
    }
    for header, body in read_documents(dataset):
        text = (corpus / header["document"]).read_text(encoding="utf-8")
        run = {(row["span"]["start"], row["span"]["end"]): row for row in run_document(text, relaxation)}
        annotated = [row for row in body if row["unit"] == "citation"]
        parsed = {row["id"]: run[anchor(row)] for row in annotated if anchor(row) in run}
        noncase = [
            (row["cited_as"]["start"], row["cited_as"]["end"], row["id"])
            for row in body
            if row["unit"] == "noncase_citation"
        ]

        for row in annotated:
            counts["citation:stated"] += 1
            by_kind[f"{row['kind']}:stated"] += 1
            found = parsed.get(row["id"])
            group = "root" if row["is_root"] else "short_form"
            counts[f"{group}:stated"] += 1
            if found is None:
                detail["citation"].append(f"{row['id']} {row['kind']} {row['cited_as']['quote'][:48]!r}")
                detail[group].append(f"{row['id']} {row['cited_as']['quote'][:48]!r} not read")
                continue
            counts["citation:right"] += 1
            by_kind[f"{row['kind']}:right"] += 1
            # A root is right when the run both finds it and reads it as one:
            # a root the run files under some other case is a root it did not
            # find, whatever it did with the span.
            if found["is_root"] == row["is_root"]:
                counts[f"{group}:right"] += 1
            else:
                reads = "a short form" if row["is_root"] else "a root"
                detail[group].append(f"{row['id']} {row['cited_as']['quote'][:44]!r} read as {reads}")

            # Attribution is scored over the citations a run actually found: a
            # citation nobody read was not attributed wrongly, it was missed,
            # and charging it here would count one failure twice.
            if found is not None:
                counts["attribution:stated"] += 1
                if found["root"] is not None:
                    counts["attribution:reported"] += 1
                expected = parsed.get(row["root_id"])
                if expected is not None and found["root"] == expected["span"]:
                    counts["attribution:right"] += 1
                elif found["root"] is not None:
                    detail["attribution"].append(f"{row['id']} attributed away from {row['root_id']}")
                else:
                    detail["attribution"].append(f"{row['id']} left unattributed, states {row['root_id']}")

            want = row.get("pin_cite")
            got = found["pin_cite"] if found else None
            if want is not None:
                counts["pincite:stated"] += 1
            if got is not None:
                counts["pincite:reported"] += 1
            if want is None:
                if got is not None:
                    detail["pincite"].append(f"{row['id']} reads a pin cite the filing does not state")
                continue
            if got is None:
                detail["pincite"].append(f"{row['id']} {want['quote']!r} not read")
            elif (got["start"], got["end"]) != (want["start"], want["end"]):
                detail["pincite"].append(f"{row['id']} {want['quote']!r} read at another span")
            elif got["pages"] != want["normalized"]:
                detail["pincite"].append(f"{row['id']} {want['quote']!r} reads as {got['pages']}")
            else:
                counts["pincite:right"] += 1

        claimed = {(row["span"]["start"], row["span"]["end"]) for row in parsed.values()}
        for span, reported in run.items():
            counts["citation:reported"] += 1
            if reported["is_root"]:
                counts["root:reported"] += 1
            else:
                counts["short_form:reported"] += 1
            if span in claimed:
                continue
            row = next((r for start, end, r in noncase if start <= span[0] and span[1] <= end), None)
            where = f"{row}, which is not a case" if row else "no annotated citation here"
            detail["citation"].append(
                f"{header['document'][:3]} {reported['kind']} {text[span[0] : span[1]][:44]!r} {where}"
            )
    counts.update(by_kind)
    return counts, detail


def _rate(right: int, of: int) -> str:
    return f"{right / of:6.1%}" if of else "     --"


def report(counts: Counter[str], detail: dict[str, list[str]], kinds: list[str]) -> str:
    """A line per measure, recall beside precision, and the disagreements under it."""
    lines = [f"{'':<18}{'recall':>18}{'precision':>18}"]
    for label, key, indent in (
        ("citations", "citation", ""),
        ("roots", "root", "- "),
        ("short forms", "short_form", "- "),
        ("pin cites", "pincite", ""),
        ("attribution", "attribution", ""),
    ):
        right = counts[f"{key}:right"]
        stated = counts[f"{key}:stated"]
        reported = counts[f"{key}:reported"]
        lines.append(
            f"{indent + label:<18}"
            f"{f'{right}/{stated}':>10}{_rate(right, stated):>8}"
            f"{f'{right}/{reported}':>10}{_rate(right, reported):>8}"
        )
    lines.append("")
    lines.append("by kind, recall:")
    lines += [f"- {kind:<22}{counts[f'{kind}:right']}/{counts[f'{kind}:stated']}" for kind in kinds]
    for label, key in (
        ("citations", "citation"),
        ("roots", "root"),
        ("short forms", "short_form"),
        ("pin cites", "pincite"),
        ("attribution", "attribution"),
    ):
        if detail[key]:
            lines.append("")
            lines.append(f"{label}:")
            lines += [f"- {line}" for line in detail[key]]
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
    kinds = sorted({key.split(":")[0] for key in counts if key.endswith(":stated") and key[0].isupper()})
    print(f"== {relaxation.name} ==")
    print(report(counts, detail, kinds))


if __name__ == "__main__":
    main()
