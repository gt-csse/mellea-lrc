"""Re-cut false-citation-bench-plus onto two labels, one entry per cited authority.

The published dataset answers one question about each cited authority: is
anything wrong with it, and which of the two things that can be wrong.

    WRONG_IDENTITY   the locator identifies no case, or identifies a case whose
                     fields disagree with the filing, or a different case sits
                     at the page
    WRONG_PINPOINT   the locator identifies one case and the fields agree, so
                     identity is sound and what is wrong is what the case is
                     cited for

**A refuted identity carries no pinpoint label.** If the case is not there, no
claim about what it says can be evaluated, so a court that named both records
one entry labelled `WRONG_IDENTITY` with the pinpoint finding kept in `comment`.
That is a rule about evidence, not a preference: the two labels have to be
disjoint or a reader cannot tell what a count means.

**One cited authority in one filing is one data point.** A filing that draws four
wrong propositions from one case, citing it in full once and returning to it
three times, is one `WRONG_PINPOINT` carrying four pieces of evidence. Grouping
by authority rather than by citation string is what makes that possible, and the
grouping comes from this project's own citation tree rather than from string
matching: `556 F.3d 177` and `556 F.3d at 201` name one case and eyecite's
resolver is what knows it.

**Provenance is a type, not an assumption.** Every entry says what document its
evidence was quoted from and who wrote it: a judge's order, an appendix in the
court's own voice, or a party's brief. Party filings are kept -- a brief naming
a fabricated citation is often the first document to name one -- but a reader
counting findings has to be able to tell an accusation from an adjudication, so
`source.adjudicated` says which it is.

The source's four kinds fold in as their evidence:

    non_existent            -> WRONG_IDENTITY
    wrong_pincite           -> WRONG_PINPOINT
    misrepresented_holding  -> WRONG_PINPOINT
    fabricated_quote        -> WRONG_PINPOINT

`annotations.json` beside the corpus is the source of record: one entry per
finding, as a court or a party wrote it. `dataset.json` is what this builds from
it, and is overwritten on every run.

    uv run python -m scripts.build_public_bench
    uv run python -m scripts.build_public_bench --out data/runs/public-bench/dataset.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
from collections import Counter
from pathlib import Path

from mellea_lrc.extraction import Relaxation, extract_from_plain_text

SOURCE = Path("data/false-citation-bench-plus")

COURT_ORDER = "court_order"
COURT_EXHIBIT = "court_exhibit"
PARTY_FILING = "party_filing"

# Who wrote the document a finding is quoted from, decided by reading each one.
# A judge's signature block settles it; where both a judge and counsel sign, the
# closing signature is the author and the other is quoted or captioned.
#
# The distinction is recorded rather than used to exclude anything. A party's
# brief naming a fabricated citation is evidence worth having -- it is often the
# first document to name one, and the court that later rules is ruling on it --
# but it is an accusation, and a reader counting findings has to be able to tell
# an accusation from an adjudication.
SOURCE_TYPES: dict[str, str] = {
    "68658788_462710593": COURT_EXHIBIT,
    "62980057_439813347": PARTY_FILING,
    "67272743_433616756": PARTY_FILING,
    "69393999_460258234": PARTY_FILING,
    "69412014_446417376": PARTY_FILING,
    "69713591_440567315": PARTY_FILING,
    "70655948_457426024": PARTY_FILING,
}
"""Every document that is not a judge's own order. The other 41 are."""

GRANTED_BY_COURT = frozenset({"69412014_446417376"})
"""A party filing the same document's order granted.

`69412014_446417376` is a one-paragraph order granting a Rule 11 motion with the
movant's brief attached as Exhibit A. The findings are quoted from the brief --
the one that resolves sits 80,000 characters before the order begins -- so the
words are counsel's, and the court adopted the motion they support.
"""

SOURCE_NOTES: dict[str, str] = {
    "68658788_462710593": (
        "An exhibit, not an order: a table headed 'Fabricated Case / Plaintiff's Use / "
        "Court's Research', filed as Doc 79-1. The third column is in the court's voice. "
        "The order it was attached to is not in this corpus."
    ),
}


def _source(document: str) -> dict[str, object]:
    """What a finding was quoted from, and whether it was adjudicated."""
    kind = SOURCE_TYPES.get(document, COURT_ORDER)
    source: dict[str, object] = {
        "document": document,
        "type": kind,
        "adjudicated": kind != PARTY_FILING,
    }
    if document in GRANTED_BY_COURT:
        source["granted_by_court"] = True
    if document in SOURCE_NOTES:
        source["note"] = SOURCE_NOTES[document]
    return source


IDENTITY_KINDS = frozenset({"non_existent"})
PINPOINT_KINDS = frozenset({"wrong_pincite", "misrepresented_holding", "fabricated_quote"})
WRONG_IDENTITY = "WRONG_IDENTITY"
WRONG_PINPOINT = "WRONG_PINPOINT"
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def _label(kind: str) -> str:
    return WRONG_IDENTITY if kind in IDENTITY_KINDS else WRONG_PINPOINT


def _fallback_key(finding: dict) -> str:
    """A grouping key for a finding no extracted citation covers.

    Four of the source's findings carry no span at all, and a span can also fall
    where extraction read nothing. Those group by the citation as the filing
    printed it, reduced to its characters, which is weaker than the tree but
    never merges two authorities that the tree would keep apart.
    """
    written = finding.get("reporter_citation") or finding.get("cited_authority") or ""
    return "written:" + _NON_ALPHANUMERIC.sub("", written.lower())


def _same_case(findings: list[dict]) -> str:
    """The case name the source records, reduced for comparison."""
    return _NON_ALPHANUMERIC.sub("", (findings[0].get("cited_authority") or "").lower())


def _merge_by_case_name(grouped: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Fold together groups the tree left apart that name one case.

    The tree merges a short form onto the full citation that introduced it, so
    `556 F.3d at 201` joins `556 F.3d 177`. It cannot do that when the filing
    never gives the case in full: `Dow AgroSciences, 637 F.3d at 268` and
    `... at 269` are two orphan short forms and become two authorities, which
    would publish one case as two data points.

    Merging on the case name the source already records closes that, and closes
    it only where the source itself says the two are the same case. Groups with
    no name are left alone rather than merged into one nameless heap.
    """
    merged: dict[str, list[dict]] = {}
    seen: dict[str, str] = {}
    for key, findings in grouped.items():
        name = _same_case(findings)
        if name and name in seen:
            merged[seen[name]].extend(findings)
            continue
        if name:
            seen[name] = key
        merged[key] = list(findings)
    return merged


def _authorities(text: str) -> tuple[dict[tuple[int, int], str], dict[str, dict]]:
    """Map each extracted citation's span to its authority, and describe each one."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
    spans: dict[tuple[int, int], str] = {}
    described: dict[str, dict] = {}
    by_id = {item.citation_id: item for item in document.citations}
    for item in document.citations:
        authority = item.authority_id or item.citation_id
        spans[(item.full_span.start, item.full_span.end)] = authority
        root = by_id.get(authority, item)
        described.setdefault(
            authority,
            {
                "citation": root.matched_text,
                "span": {"start": root.locator_span.start, "end": root.locator_span.end},
            },
        )
    return spans, described


def _covering(spans: dict[tuple[int, int], str], start: int, end: int) -> str | None:
    """The authority of the extracted citation that covers this span, if one does."""
    best: tuple[int, str] | None = None
    for (a, b), authority in spans.items():
        if a <= start and end <= b:
            width = b - a
            if best is None or width < best[0]:
                best = (width, authority)
    return None if best is None else best[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=None, help="default: overwrite the source's dataset.json")
    args = parser.parse_args()

    source = json.loads((args.source / "annotations.json").read_text(encoding="utf-8"))
    filings = (
        source["documents"]
        if "documents" in source
        else next(
            value
            for key, value in source.items()
            if isinstance(value, list)
            and value
            and isinstance(value[0], dict)
            and "false_citations" in value[0]
        )
    )

    entries: list[dict] = []
    counts: Counter = Counter()
    for filing in filings:
        text = (args.source / filing["filing_text"]).read_text(encoding="utf-8", errors="replace")
        spans, described = _authorities(text)

        grouped: dict[str, list[dict]] = {}
        for finding in filing["false_citations"]:
            found = None
            for span in finding.get("spans") or []:
                found = _covering(spans, span["start"], span["end"])
                if found is not None:
                    break
            key = found or _fallback_key(finding)
            counts["grouped by the citation tree" if found else "grouped by the printed citation"] += 1
            grouped.setdefault(key, []).append(finding)

        before = len(grouped)
        grouped = _merge_by_case_name(grouped)
        counts["groups merged because the source names one case"] += before - len(grouped)

        for key, findings in grouped.items():
            labels = {_label(item["kind"]) for item in findings}
            label = WRONG_IDENTITY if WRONG_IDENTITY in labels else WRONG_PINPOINT
            counts[label] += 1
            comment = ""
            if len(labels) > 1:
                counts["identity refuted a citation a court also faulted on its pinpoint"] += 1
                also = sorted({item["kind"] for item in findings if item["kind"] in PINPOINT_KINDS})
                comment = (
                    f"The court also faulted this citation on its pinpoint ({', '.join(also)}). "
                    f"No pinpoint label is recorded: the case is not at the locator, so nothing "
                    f"it is cited for can be evaluated. The finding is kept as evidence below."
                )
            # Two findings about one authority often quote the same citation, so
            # the spans repeat. An occurrence is a place in the filing, not a
            # finding about one.
            occurrences = []
            placed: set[tuple[int, int]] = set()
            for item in findings:
                for span in item.get("spans") or []:
                    where = (span["start"], span["end"])
                    if where not in placed:
                        placed.add(where)
                        occurrences.append(span)
            occurrences.sort(key=lambda span: span["start"])
            entries.append(
                {
                    "filing": filing["filing"],
                    "case_name": filing["case_name"],
                    "court_id": filing["court_id"],
                    "source": _source(filing["source_document"]),
                    "split": filing["split"],
                    "label": label,
                    "cited_authority": findings[0]["cited_authority"],
                    "reporter_citation": findings[0]["reporter_citation"],
                    "authority": described.get(key),
                    "occurrences": occurrences,
                    "comment": comment,
                    "evidence": [
                        {
                            "source_kind": item["kind"],
                            "proposition": item.get("proposition") or "",
                            "quote": item.get("quote") or "",
                            "ruling_evidence": item["ruling_evidence"],
                        }
                        for item in findings
                    ],
                }
            )

    dataset = {
        "name": source.get("name", "false-citation-bench-plus"),
        "schema": "mellea-lrc/false-citation/two-label/v1",
        "unit": "one cited authority in one filing",
        "labels": [WRONG_IDENTITY, WRONG_PINPOINT],
        "ground_truth": source.get("ground_truth"),
        "filing_count": len(filings),
        "entry_count": len(entries),
        "label_counts": {WRONG_IDENTITY: counts[WRONG_IDENTITY], WRONG_PINPOINT: counts[WRONG_PINPOINT]},
        "source_counts": dict(sorted(Counter(entry["source"]["type"] for entry in entries).items())),
        "entries": entries,
    }
    out = args.out or (args.source / "dataset.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataset, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{len(filings)} filings, {sum(len(f['false_citations']) for f in filings)} findings")
    print(f"-> {len(entries)} entries at {out}")
    for label, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {label:<58}{count:>5}")
    print("  --")
    for kind, count in sorted(dataset["source_counts"].items()):
        print(f"  {kind:<58}{count:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
