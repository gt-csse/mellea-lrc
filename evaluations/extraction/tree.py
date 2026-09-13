"""Score extraction against the citation tree ground truth, arm by arm.

The bench `evaluate.py` reads is a flat list of identifiers: it asks whether a
citation was found and nothing else. This one reads `extraction-v3.0`, which is
a tree -- every place a filing cites a case, which place introduced the case,
and which page each one claims -- and asks the questions separately, because a
pass that finds every citation and files half of them under the wrong case is
not doing better than one that finds fewer.

## The arms

``eyecite``
    eyecite as published. The floor, and what a result is read up from.

``augmented``
    the same, with this project's rules: the separator relaxation that matches
    the whitespace PDF extraction leaves, the docket reader, the pin cite
    reader, and the case name locator.

``mellea``
    the augmented rules, and then the model layers, each of which answers one
    question the rules leave. Today that is the case-name layer, which sweeps
    the residue of a full mask and says what each case name standing in it is.
    More layers land here as they are built.

## What is scored, and how

An annotated citation is **found** when the arm produces a citation at exactly
its `locator` span, or where it starts when it states no locator, which is `Id.`
and the bare-name references. Nothing partial counts: `PROTOCOL.md` fixes that
rule, and the identifier is what a lookup resolves.

The **roots** are counted apart from the **short forms**, because the two cost
different things: a short form missed costs a page claim, a root missed costs
the case. A root counts as found only when the arm reads it *as* a root -- a
root filed under some other case is a root it did not find, whatever it did
with the span.

A root is named by its span rather than by an id, so an arm that knows nothing
about this dataset's ids can still be scored: a citation is **attributed** when
the arm's root sits at the span of the row the ground truth calls its root.
Attribution is scored over the citations an arm found, because a citation
nobody read was missed rather than misattributed, and charging it here would
count one failure twice.

**Recall is out of what the filings state; precision is out of what the arm
reports.** One without the other hides half of a pass: a reader that reports
every span in the document has perfect recall. Both denominators are taken over
the whole run rather than over the part of it the dataset annotates -- a pin
cite stays in the recall denominator when the citation carrying it was missed,
and an attribution counts against precision even when the thing attributed is
not a citation to a case.

Statutes and journal citations are not in this ground truth and are not counted
either way. A statute *read as a case* is a case-kind citation at a span no row
claims, so it costs precision, which is where it belongs.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mellea_lrc.core.case_names import CaseName
from mellea_lrc.core.citations import CitationKind, citation_kind
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.adjudication.candidates.case_name_sites import case_name_sites
from mellea_lrc.extraction.adjudication.review.case_name import Reading, adjudicate_case_name
from mellea_lrc.llm import start_mellea_session_from_env

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.extraction.types import ExtractedDocument

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

CASE_NAME_LAYER = "case_name"


@dataclass(frozen=True, slots=True)
class Arm:
    """One pass over a document, named for what it runs."""

    relaxation: Relaxation
    layers: tuple[str, ...] = field(default_factory=tuple)
    components: str = ""
    reads_dockets: bool = True
    """Whether docket numbers are read at all.

    eyecite attempts none: its tokenizer is built from a reporter gazetteer and
    a docket number names no reporter. The reader that finds them is this
    project's, so an arm standing for eyecite as published must not have it,
    and the 42 docket citations in this corpus are 42 citations it cannot see.
    """

    @property
    def needs_model(self) -> bool:
        """Whether running it calls a model, and so costs time and money."""
        return bool(self.layers)


ARMS = {
    "eyecite": Arm(Relaxation.NONE, components="eyecite as published", reads_dockets=False),
    "augmented": Arm(Relaxation.FULL, components="+ this project's rules"),
    "mellea": Arm(Relaxation.FULL, (CASE_NAME_LAYER,), components="+ the case-name layer"),
}

MEASURES = (
    ("citations", "citation", ""),
    ("roots", "root", "- "),
    ("short forms", "short_form", "- "),
    ("pin cites", "pincite", ""),
    ("docket courts", "court", ""),
    ("attribution", "attribution", ""),
)


def anchor(row: dict[str, Any]) -> tuple[int, int]:
    """The span an annotated citation is matched at.

    The locator, or the whole citation where the row states no locator: `Id.`
    and a bare name identify nothing, which is what makes them short forms.
    """
    span = row.get("locator") or row["cited_as"]
    return (span["start"], span["end"])


#: What a citation that identifies nothing is matched on: where it starts.
_NO_LOCATOR = -1
_UNLOCATED = {CitationKind.ID.value, CitationKind.REFERENCE.value}


def identity(row: dict[str, Any]) -> tuple[int, int]:
    """The span an annotated citation is *identified* at.

    The locator, or the start alone where the row states none. `Id.   at 71
    n.10` is one citation and `cited_as` covers all of it, but how much of a
    pin cite written past the `Id.` a pass reads is the pin cite's own row to
    answer, and charging it here would count one failure twice.
    """
    locator = row.get("locator")
    if locator is not None:
        return (locator["start"], locator["end"])
    return (row["cited_as"]["start"], _NO_LOCATOR)


def identities(run: dict[tuple[int, int], dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    """What the run reports, keyed the way `identity` keys an annotated row."""
    return {
        (span[0], _NO_LOCATOR) if reported["kind"] in _UNLOCATED else span: reported
        for span, reported in run.items()
    }


def read_documents(dataset: Path) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Each document's header row and its annotations."""
    for path in sorted(dataset.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        yield rows[0], rows[1:]


def _row(
    kind: str,
    span: tuple[int, int],
    root: tuple[int, int] | None,
    pin_cite: Any,
    court: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "span": {"start": span[0], "end": span[1]},
        "root": {"start": root[0], "end": root[1]} if root else None,
        "is_root": root is not None and root == span,
        "pin_cite": pin_cite,
        "court": court,
    }


def _overlaps(one: tuple[int, int], other: tuple[int, int]) -> int:
    """How many characters two spans share."""
    return max(0, min(one[1], other[1]) - max(one[0], other[0]))


def _associate(
    run: dict[tuple[int, int], dict[str, Any]], annotated: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Which reported citation is which annotated one, by the span they share.

    A row is associated with the reported citation that overlaps it most, and a
    reported citation is associated with at most one row: two annotated
    citations at one span would be the same citation twice.
    """
    taken: set[tuple[int, int]] = set()
    found: dict[str, dict[str, Any]] = {}
    for row in annotated:
        want = anchor(row)
        best = max(
            (span for span in run if span not in taken and _overlaps(span, want)),
            key=lambda span: (_overlaps(span, want), -abs(span[0] - want[0])),
            default=None,
        )
        if best is not None:
            taken.add(best)
            found[row["id"]] = run[best]
    return found


def _from_rules(extracted: ExtractedDocument, *, dockets: bool) -> dict[tuple[int, int], dict[str, Any]]:
    """What the deterministic pass reports, keyed by the span it reports it at."""
    at = {citation.citation_id: citation.locator_span for citation in extracted.citations}
    rows = {}
    for citation in extracted.citations:
        kind = citation_kind(citation.citation)
        if kind not in CASE_KINDS or (kind is CitationKind.DOCKET and not dockets):
            continue
        root = at.get(citation.root_id or "")
        span = (citation.locator_span.start, citation.locator_span.end)
        pin = citation.pin_cite_span
        rows[span] = _row(
            citation_kind(citation.citation).value,
            span,
            (root.start, root.end) if root else None,
            {
                "start": pin.start,
                "end": pin.end,
                "pages": [
                    {"first": page.first, "last": page.last, "kind": page.kind.value}
                    for page in citation.pin_cite_pages
                ],
            }
            if pin
            else None,
            getattr(citation.citation, "court", None)
            if citation_kind(citation.citation) is CitationKind.DOCKET
            else None,
        )
    return rows


async def _case_name_layer(
    extracted: ExtractedDocument, rows: dict[tuple[int, int], dict[str, Any]], session: MelleaSession
) -> None:
    """Apply what a reader makes of each case name standing outside every citation.

    `short_form` adds a row: a citation the rules did not read at all. The other
    three change no row. `names_a_citation` writes the name onto the citation it
    names, which this table does not score but which the record keeps, so the
    reader's answer stops being thrown away; the last two are findings about the
    filing rather than citations.
    """
    at = {citation.citation_id: citation.locator_span for citation in extracted.citations}
    by_id = {citation.citation_id: citation for citation in extracted.citations}
    for site in case_name_sites(extracted):
        answer = await adjudicate_case_name(extracted, site, session=session)
        if answer is None:
            continue
        if answer.reading is Reading.NAMES_A_CITATION:
            named = by_id.get(answer.citation_id or "")
            if named is not None:
                named.record_case_name(
                    CaseName(
                        span=answer.span,
                        text=answer.name,
                        plaintiff=answer.plaintiff,
                        defendant=answer.defendant,
                    ),
                    by=CASE_NAME_LAYER,
                    reason=answer.reason or None,
                )
            continue
        if answer.reading is not Reading.SHORT_FORM:
            continue
        root = at.get(answer.root_id or "")
        if root is None:
            continue
        span = (answer.span.start, answer.span.end)
        rows[span] = _row("ReferenceCitation", span, (root.start, root.end), None)


async def run_document(
    text: str, arm: Arm, session: MelleaSession | None
) -> dict[tuple[int, int], dict[str, Any]]:
    """Run one arm over one document and project what it reports."""
    # Eyecite writes overlap diagnostics to stdout as it reads.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        extracted = extract_from_plain_text(text, relaxation=arm.relaxation)
    rows = _from_rules(extracted, dockets=arm.reads_dockets)
    if CASE_NAME_LAYER in arm.layers and session is not None:
        await _case_name_layer(extracted, rows, session)
    return rows


async def score(dataset: Path, corpus: Path, arm: Arm) -> tuple[Counter[str], dict[str, list[str]]]:
    """Score one arm over the whole corpus."""
    counts: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    detail: dict[str, list[str]] = {key: [] for _, key, _ in MEASURES}
    session = start_mellea_session_from_env() if arm.needs_model else None
    for header, body in read_documents(dataset):
        text = (corpus / header["document"]).read_text(encoding="utf-8")
        run = await run_document(text, arm, session)
        annotated = [row for row in body if row["unit"] == "citation"]
        roots_by_id = {row["id"]: row.get("identifier") or {} for row in annotated if row["is_root"]}
        reported = identities(run)
        parsed = {row["id"]: reported[identity(row)] for row in annotated if identity(row) in reported}
        # Identification is exact, because the identifier is what a lookup
        # resolves. Everything scored *about* a citation is scored over an
        # overlap instead: a run that reads `673 F.2d at ` where the filing
        # writes `673 F.2d at 57` has the citation, with the wrong edges, and
        # whether it then attributed it correctly is a separate question.
        associated = _associate(run, annotated)
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
            # Counted before the citation is looked for, so a pin cite stays in
            # the denominator when the citation carrying it was missed.
            want = row.get("pin_cite")
            if want is not None:
                counts["pincite:stated"] += 1
            want_court = (roots_by_id.get(row["root_id"]) or {}).get("court")
            if row["kind"] == "DocketCitation" and want_court:
                counts["court:stated"] += 1

            # **Attribution is a short form pointing at its root**, so it is
            # counted over the short forms the filings write. A root points at
            # itself and there is nothing there to get wrong.
            if not row["is_root"]:
                counts["attribution:stated"] += 1
                reported = associated.get(row["id"])
                target = associated.get(row["root_id"])
                if reported is None:
                    detail["attribution"].append(f"{row['id']} not read, so it points nowhere")
                elif reported["root"] is None:
                    detail["attribution"].append(f"{row['id']} left unattributed, states {row['root_id']}")
                elif target is not None and _overlaps(
                    (reported["root"]["start"], reported["root"]["end"]),
                    (target["span"]["start"], target["span"]["end"]),
                ):
                    counts["attribution:right"] += 1
                else:
                    detail["attribution"].append(f"{row['id']} attributed away from {row['root_id']}")

            at_span = associated.get(row["id"])
            if row["kind"] == "DocketCitation" and want_court:
                if at_span is None:
                    detail["court"].append(f"{row['id']} not read, so it names no court")
                elif at_span["court"] == want_court:
                    counts["court:right"] += 1
                else:
                    detail["court"].append(
                        f"{row['id']} read the court as {at_span['court']!r}, and the filing "
                        f"writes one that resolves to {want_court!r}"
                    )
            if found is None:
                detail["citation"].append(f"{row['id']} {row['kind']} {row['cited_as']['quote'][:48]!r}")
                detail[group].append(f"{row['id']} {row['cited_as']['quote'][:48]!r} not read")
                if want is not None:
                    detail["pincite"].append(f"{row['id']} {want['quote']!r} not read, nor was its citation")
                continue
            counts["citation:right"] += 1
            by_kind[f"{row['kind']}:right"] += 1
            if found["is_root"] == row["is_root"]:
                counts[f"{group}:right"] += 1
            else:
                reads = "a short form" if row["is_root"] else "a root"
                detail[group].append(f"{row['id']} {row['cited_as']['quote'][:44]!r} read as {reads}")

            got = found["pin_cite"]
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
            counts["root:reported" if reported["is_root"] else "short_form:reported"] += 1
            if reported["root"] is not None and not reported["is_root"]:
                counts["attribution:reported"] += 1
            if reported["pin_cite"] is not None:
                counts["pincite:reported"] += 1
            if reported["court"] is not None:
                counts["court:reported"] += 1
            if span in claimed:
                continue
            row = next((r for start, end, r in noncase if start <= span[0] and span[1] <= end), None)
            where = f"{row}, which is not a case" if row else "no annotated citation here"
            detail["citation"].append(
                f"{header['document'][:3]} {reported['kind']} {text[span[0] : span[1]][:44]!r} {where}"
            )
            if reported["pin_cite"] is not None:
                pin = reported["pin_cite"]
                detail["pincite"].append(
                    f"{header['document'][:3]} {text[pin['start'] : pin['end']]!r} claimed at "
                    f"{text[span[0] : span[1]][:32]!r}, which is not a citation to a case"
                )
            if reported["root"] is not None:
                root = text[reported["root"]["start"] : reported["root"]["end"]]
                detail["attribution"].append(
                    f"{header['document'][:3]} {text[span[0] : span[1]][:32]!r} attributed to "
                    f"{root!r}, and it is not a citation to a case"
                )
    counts.update(by_kind)
    return counts, detail


def _cell(counts: Counter[str], key: str, *, as_rate: bool) -> str:
    """One arm's reading of one measure: recall beside precision.

    Each side is `--` on its own when its denominator is empty, which is not
    the same failure: an arm that reports nothing of a kind has no precision to
    state and a recall of zero.
    """
    right, stated, reported = (counts[f"{key}:{part}"] for part in ("right", "stated", "reported"))
    if not as_rate:
        return f"{right}/{stated} · {right}/{reported}"
    recall = f"{right / stated:.1%}" if stated else "--"
    precision = f"{right / reported:.1%}" if reported else "--"
    return f"{recall} · {precision}"


def table(scores: dict[str, Counter[str]], *, as_rate: bool) -> str:
    """One table over every arm, a row per measure, each cell recall · precision."""
    width = max(19, *(len(name) + 2 for name in scores))
    lines = ["".ljust(14) + "".join(name.ljust(width) for name in scores)]
    for label, key, indent in MEASURES:
        lines.append(
            (indent + label).ljust(14)
            + "".join(_cell(counts, key, as_rate=as_rate).ljust(width) for counts in scores.values())
        )
    return "\n".join(lines)


def kinds_table(scores: dict[str, Counter[str]]) -> str:
    """Recall per citation kind, which is where a miss says what shape it was."""
    kinds = sorted(
        {key.split(":")[0] for counts in scores.values() for key in counts if key.endswith(":stated")}
        - {label for _, label, _ in MEASURES}
    )
    width = max(19, *(len(name) + 2 for name in scores))
    lines = ["".ljust(24) + "".join(name.ljust(width) for name in scores)]
    for kind in kinds:
        lines.append(
            f"- {kind}".ljust(24)
            + "".join(
                f"{counts[f'{kind}:right']}/{counts[f'{kind}:stated']}".ljust(width)
                for counts in scores.values()
            )
        )
    return "\n".join(lines)


def main() -> None:
    """Run the command-line tree evaluator."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="arms:\n" + "\n".join(f"  {name:<12} {arm.components}" for name, arm in ARMS.items()),
    )
    parser.add_argument("--dataset", type=Path, required=True, help="extraction-v3.0/documents/")
    parser.add_argument("--documents", type=Path, required=True, help="The text those spans index.")
    parser.add_argument(
        "--arms", nargs="+", default=list(ARMS), choices=list(ARMS), help="Which arms to run."
    )
    parser.add_argument("--detail", action="store_true", help="Print every disagreement.")
    args = parser.parse_args()

    scores: dict[str, Counter[str]] = {}
    details: dict[str, dict[str, list[str]]] = {}
    for name in args.arms:
        arm = ARMS[name]
        if arm.needs_model:
            print(f"{name}: calls a model; set MELLEA_LRC_LLM_* in the environment")
        counts, detail = asyncio.run(score(args.dataset, args.documents, arm))
        scores[name], details[name] = counts, detail

    print("\ncounts, recall · precision\n")
    print(table(scores, as_rate=False))
    print("\npercentages, recall · precision\n")
    print(table(scores, as_rate=True))
    print("\nby kind, recall\n")
    print(kinds_table(scores))
    if args.detail:
        for name, detail in details.items():
            for label, key, _ in MEASURES:
                if detail[key]:
                    print(f"\n{name} · {label}")
                    print("\n".join(f"- {line}" for line in detail[key]))


if __name__ == "__main__":
    main()
