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

An `out_of_scope_citation` row is different, and is counted in **neither**
direction. It marks a case the filing locates by something this dataset does not
score -- an agency's own file number, a slip opinion. There are without limit
many of those, and a ground truth that scores each new one as it appears grows
by whatever the next filing invents; so the row records what the filing wrote
and an arm that reads it is neither credited nor charged.

## The bare name is out of scope, for now

A name-only `ReferenceCitation` is left out of both sides by default: out of
every recall denominator, and out of precision wherever an arm reports one.
Rule 10.9 lets a filing write a case name and stop, and finding one is a
question about case names rather than about citations -- nothing is written at
that position for a citation reader to reach, and the answer comes after the
citations are read and the roots are resolved. Scoring a question the pipeline
does not yet ask says nothing about the pipeline. `--score-bare-names` puts
them back.

A reference that states a page -- `Bell at 546` -- is always scored. The page is
a claim about the opinion, and every other citation's page is scored.

The dataset is untouched: these stay ordinary `citation` rows, because a bare
name is a conforming citation and the ground truth says what the filing wrote.
Leaving them out is the evaluator's decision about what is being measured
today, not a claim about the citation.

`--replay` scores a run already written by `--artifacts` instead of producing
one, so an arm that calls a model can be re-scored against a changed ground
truth without calling it again.
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

from evaluations.extraction.identity import settled
from mellea_lrc.core.citations import CitationKind, citation_kind
from mellea_lrc.core.pin_cites import PinCiteKind
from mellea_lrc.extraction import Attachment, Relaxation, extract_from_plain_text, grow_leaves
from mellea_lrc.extraction.adjudication import Review, adjudicate
from mellea_lrc.llm import start_mellea_session_from_env
from mellea_lrc.serialization import (
    deserialize_document,
    serialize_document,
)

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.extraction.types import Document

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

@dataclass(frozen=True, slots=True)
class Arm:
    """One pass over a document, named for what it runs."""

    relaxation: Relaxation
    reviews: tuple[Review, ...] = field(default_factory=tuple)
    components: str = ""
    needs_identity: bool = False
    """Whether the roots are handed to validation's identity stage first.

    Replayed from a recorded run rather than called, so the arm costs nothing
    to score. See :mod:`evaluations.extraction.identity`.
    """

    attach: Attachment = Attachment.STATED
    """How the leaves are attached to their roots once the roots are settled.

    `Attachment.EYECITE` is eyecite's own resolution, decided while parsing
    against the party names the parser read. `Attachment.STATED` is this
    project's second growth, decided from `stated` -- the citation as it is
    after the readers and validation have corrected it. The two arms differ in
    nothing else, so a difference between them is the attachment.
    """
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
        return bool(self.reviews)


ARMS = {
    "eyecite": Arm(
        Relaxation.NONE,
        components="eyecite as published",
        reads_dockets=False,
        attach=Attachment.EYECITE,
    ),
    "augmented": Arm(
        Relaxation.FULL, components="+ this project's rules", attach=Attachment.EYECITE
    ),
    "grown": Arm(Relaxation.FULL, components="+ leaves grown from `stated`"),
    # `Review.CASE_NAME` is off, and so is every other search for a citation
    # nobody read. Both turn on a name, and the names in the record here are
    # whatever eyecite's parser made of them. They come back after validation
    # has resolved the roots, when the record holds each authority's real name.
    # `--defer-bare-names` is the scoring side of the same decision: the arm is
    # not asked for them and the score does not count them.
    "mellea": Arm(
        Relaxation.FULL, (Review.PIN_CITE,), components="+ the pin-cite review, then the leaves"
    ),
    "grown+identity": Arm(
        Relaxation.FULL,
        components="+ leaves grown over the roots identity settled",
        needs_identity=True,
    ),
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


def _from_rules(extracted: Document, *, dockets: bool) -> dict[tuple[int, int], dict[str, Any]]:
    """What the deterministic pass reports, keyed by the span it reports it at."""
    at = {citation.citation_id: citation.locator_span for citation in extracted.citations}
    rows = {}
    for citation in extracted.citations:
        kind = citation_kind(citation.stated)
        if kind not in CASE_KINDS or (kind is CitationKind.DOCKET and not dockets):
            continue
        root = at.get(citation.root_id or "")
        span = (citation.locator_span.start, citation.locator_span.end)
        pin = citation.pin_cite_span
        rows[span] = _row(
            citation_kind(citation.stated).value,
            span,
            (root.start, root.end) if root else None,
            {
                "start": pin.start,
                "end": pin.end,
                # `footnote` only where there is one, so a page claim compares
                # equal to a ground truth that states the same page and no
                # footnote.
                #
                # An arm that read no page out of the characters reports an
                # empty list, which is what the ground truth writes when the
                # characters state no page. The dataset says what the filing
                # claims; `PinCiteKind.UNREAD` is this project saying what it
                # could do with the characters, and the two are not the same
                # statement, so the second one is not compared against the
                # first -- it is dropped, and agreeing means both sides ended
                # with no page.
                "pages": [
                    {"first": page.first, "last": page.last, "kind": page.kind.value}
                    | ({"footnote": page.footnote} if page.footnote else {})
                    for page in citation.pin_cite_pages
                    if page.kind is not PinCiteKind.UNREAD
                ],
            }
            if pin
            else None,
            getattr(citation.stated, "court", None)
            if citation_kind(citation.stated) is CitationKind.DOCKET
            else None,
        )
    return rows


async def run_document(
    text: str,
    arm: Arm,
    session: MelleaSession | None,
    name: str = "",
    artifacts: Path | None = None,
    identity_run: Path | None = None,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Run one arm over one document and project what it reports.

    **Through the artifact.** What is scored is what comes back from
    `serialize_document` and `deserialize_document`, not the
    objects in memory, because the artifact is what the next stage reads. A
    field that does not survive the round trip is a field validation does not
    have, and a score taken before it would not say so.
    """
    # Eyecite writes overlap diagnostics to stdout as it reads.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        extracted = extract_from_plain_text(text, relaxation=arm.relaxation)
    if arm.reviews and session is not None:
        await adjudicate(extracted, arm.reviews, session=session)
    if arm.needs_identity and identity_run is not None:
        extracted = settled(extracted, identity_run / f"{Path(name).stem}.json")
    # The leaves last, and over the roots as they stand: this is the order the
    # pipeline runs in, where validation settles the roots between the two
    # growths and a leaf is matched against a name that has been checked.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        extracted = grow_leaves(extracted, attach=arm.attach)
    payload = serialize_document(extracted)
    if artifacts is not None and name:
        _write_artifact(artifacts, name, payload)
    return _from_rules(deserialize_document(payload), dockets=arm.reads_dockets)


def _replayed(replay: Path, name: str, *, dockets: bool) -> dict[tuple[int, int], dict[str, Any]]:
    """One arm's saved run, read back instead of produced.

    The artifact is what `run_document` scores anyway, so a replay of it scores
    the same run -- which is what lets an arm that calls a model be re-scored
    against a changed ground truth without calling it again.
    """
    payload = json.loads((replay / f"{Path(name).stem}.json").read_text(encoding="utf-8"))
    return _from_rules(deserialize_document(payload), dockets=dockets)


def _write_artifact(artifacts: Path, name: str, payload: dict[str, Any]) -> None:
    """One document's run, as the next stage will read it."""
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / f"{Path(name).stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def _is_bare_name(kind: str | None, pin_cite: object) -> bool:
    """A name-only reference: Rule 10.9's short form, with no page after it.

    `Bell at 546` is a reference too and states a page, which is a claim about
    the opinion that can be right or wrong. `In Burrell` states nothing, so
    whether it is read at all is a question about the name and not about the
    citation.
    """
    return kind == "ReferenceCitation" and not pin_cite


async def score(
    dataset: Path,
    corpus: Path,
    arm: Arm,
    artifacts: Path | None = None,
    replay: Path | None = None,
    *,
    defer_bare_names: bool = False,
    identity_run: Path | None = None,
) -> tuple[Counter[str], dict[str, list[str]]]:
    """Score one arm over the whole corpus, writing its artifacts when asked.

    ``defer_bare_names`` takes the name-only references out of both sides: out
    of every recall denominator, and out of precision wherever the arm reports
    one or reports anything at a deferred row's span. Reading a bare name is a
    question about case names, which is answered after the citations are read,
    and this says what the rest of the pass looks like without it.
    """
    counts: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    detail: dict[str, list[str]] = {key: [] for _, key, _ in MEASURES}
    session = start_mellea_session_from_env() if arm.needs_model and replay is None else None
    for header, body in read_documents(dataset):
        text = (corpus / header["document"]).read_text(encoding="utf-8")
        run = (
            _replayed(replay, header["document"], dockets=arm.reads_dockets)
            if replay is not None
            else await run_document(text, arm, session, header["document"], artifacts, identity_run)
        )
        annotated = [row for row in body if row["unit"] == "citation"]
        # Counted in nothing, in either direction. A case can be located in
        # without limit many ways -- an agency's own file number, a slip
        # opinion, a docket sheet -- and a ground truth that scores each new one
        # as it appears grows by whatever the next filing invents. So these
        # leave the recall denominator with their unit, and an arm that reads
        # one is neither credited nor charged.
        outside = [
            (row["cited_as"]["start"], row["cited_as"]["end"])
            for row in body
            if row["unit"] == "out_of_scope_citation"
        ]
        deferred: list[tuple[int, int]] = []
        if defer_bare_names:
            deferred = [
                (row["cited_as"]["start"], row["cited_as"]["end"])
                for row in annotated
                if _is_bare_name(row.get("kind"), row.get("pin_cite"))
            ]
            annotated = [
                row for row in annotated if not _is_bare_name(row.get("kind"), row.get("pin_cite"))
            ]
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
            if any(_overlaps(span, where) for where in outside):
                continue
            if defer_bare_names and (
                _is_bare_name(reported["kind"], reported["pin_cite"])
                or any(_overlaps(span, where) for where in deferred)
            ):
                continue
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
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="Write each arm's serialized run here, one file per document, under the arm's name.",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="Score the runs already written here instead of producing them, same layout as "
        "--artifacts. An arm that calls a model is re-scored without calling it.",
    )
    parser.add_argument(
        "--identity",
        type=Path,
        default=None,
        help="A recorded identity run, one file per document. The arms that grow leaves over "
        "settled roots need it; without it they are the same as `grown`.",
    )
    parser.add_argument(
        "--score-bare-names",
        action="store_true",
        help="Score the name-only ReferenceCitation rows, which are left out of both sides by "
        "default. A reference that states a page is always scored.",
    )
    args = parser.parse_args()

    scores: dict[str, Counter[str]] = {}
    details: dict[str, dict[str, list[str]]] = {}
    for name in args.arms:
        arm = ARMS[name]
        if arm.needs_model and args.replay is None:
            print(f"{name}: calls a model; set MELLEA_LRC_LLM_* in the environment")
        counts, detail = asyncio.run(
            score(
                args.dataset,
                args.documents,
                arm,
                args.artifacts / name if args.artifacts else None,
                args.replay / name if args.replay else None,
                defer_bare_names=not args.score_bare_names,
                identity_run=args.identity,
            )
        )
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
