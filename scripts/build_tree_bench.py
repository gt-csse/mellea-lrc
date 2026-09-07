r"""Build `false-citation-bench-tree-v2.0`: ground truth as a citation tree.

Every bench this project has had so far annotates **full citations**. A filing
does not cite an authority once, though. It cites it in full, then returns to it
as `Id. at 570`, `550 U.S. at 563`, or by party name, and each return visit
names a different page and attaches a different proposition to it. A bench of
full citations only can score whether the case was identified; it cannot score
the larger part of what a brief actually claims.

This bench annotates the tree instead. The unit is the **authority** -- one
cited case -- and under it every **occurrence**: the full citation that
introduced it and every short form, `Id.`, supra and party-name reference that
returns to it.

## The two annotation rules that decide everything here

**Attribution is recorded sincerely.** An occurrence is filed under the
authority it *should* belong to, read from the text, not under whatever the
current resolver produces. Where the two disagree the ground truth follows the
reading and `build-report.md` names the disagreement. A bench that recorded the
resolver's answer could never measure the resolver.

**Only what is written is recorded.** A bare `Id.` states no page; its effective
page comes from its antecedent. Recording that inherited page would bake a tree
decision into the label and make inheritance unmeasurable, so `pin_cite_written`
is null for a bare `Id.` and inheritance is left to be computed and scored.

For the same reason a pin cite is recorded **verbatim**, spacing damage
included: `998 -1003`, not `998-1003`. Normalisation is a separate step that
has to be testable, and ground truth that is pre-normalised silently encodes the
repair being evaluated.

## What counts as an occurrence

A place the document *cites* the authority: a full citation, a short form, an
`Id.`, a supra, or a party-name reference. A bare party name in prose is not an
occurrence -- sometimes a name is just a name -- so the eleven times document
022 writes "Advanced Textile" are not eleven citations.

## What is out of scope, and why it is recorded rather than dropped

An authority is a court case, however identified. A reference into the record --
`Id. ¶ 33` pointing at an indictment's numbered allegations, or `Id. ¶¶ 26-28`
pointing at a motion -- is not a case, and neither is a citation nested inside
another case's quotation. These are recorded with `scope: "out_of_scope"` and a
reason, not deleted, so that an extractor which finds them is not charged with a
false positive for being right.

    uv run python scripts/build_tree_bench.py

Everything the build could not settle goes to `build-report.md`. `data/` is
gitignored; nothing here is published.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mellea_lrc.core.citations import CitationKind, FullCaseCitation
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.structure.citation_tree import build_citation_tree
from mellea_lrc.extraction.types import ExtractedCitation

if TYPE_CHECKING:
    from collections.abc import Sequence


SECONDARY_KINDS = frozenset(
    {CitationKind.SHORT_CASE, CitationKind.ID, CitationKind.SUPRA, CitationKind.REFERENCE}
)

# A pin cite is a page of the authority just named: an optional `at`, then
# digits, star pages, paragraph or section marks, ranges and lists of those.
# Deliberately narrow -- it decides ground truth, so it reads only shapes that
# are unambiguously a pin cite and stops at the first thing that is not.
_PAGE = r"(?:\*\s*\d+|¶{1,2}\s*\d+|\d+)"
# A section sign is deliberately absent from `_PAGE` and is never skipped over
# silently. A case is paginated; a statute is sectioned. `Id. § 1231` returns to
# a statute, and reading `1231` as a page would attach a pinpoint claim about a
# case to a number that names something else entirely. Where one stands in the
# pin-cite position the occurrence is recorded out of scope with that as the
# reason, so the evidence is kept rather than dropped.
_SECTION_AHEAD = re.compile(r"[^\S\r\n]*(?:,[^\S\r\n]*)?§")
_RANGE = rf"{_PAGE}(?:[^\S\r\n]*[-–][^\S\r\n]*\d+)?"
PIN_FROM_TEXT = re.compile(
    rf"""
    [^\S\r\n]*(?:,[^\S\r\n]*)?          # an optional comma after the identifier
    (?P<pin>
        (?:at[^\S\r\n]+)?               # `at`, as short forms and `Id.` write it
        {_RANGE}
        (?:[^\S\r\n]*,[^\S\r\n]*{_RANGE})*   # further pages in the same cite
        (?:[^\S\r\n]*n\.[^\S\r\n]*\d+)?      # a trailing footnote
    )
    """,
    re.VERBOSE,
)

# Where a citation writes its page inside its own span rather than after it.
# eyecite extends the span over a pin cite it managed to parse and stops at the
# signal word when it did not, so `id. at 1447` and `Id.` are both spelled
# `Id.`-plus-a-page in the document and differ only in whether the parse
# succeeded. Scanning from the `at` reads both the same way.
_PIN_INSIDE = {
    # `556 U.S. at 678`: the volume is a number too, so only `at` marks the page.
    CitationKind.SHORT_CASE: re.compile(r"\bat\b"),
    CitationKind.ID: re.compile(r"\bat\b|[*\u00b6]|\d"),
    CitationKind.SUPRA: re.compile(r"\bat\b|[*\u00b6]|\d"),
    CitationKind.REFERENCE: re.compile(r"\bat\b|[*\u00b6]|\d"),
}


@dataclass(frozen=True)
class Decision:
    """One annotation made by reading the text, overriding what the tree produced."""

    action: str  # "reattribute" | "out_of_scope" | "no_authority"
    why: str
    to_locator: tuple[str, str, str] | None = None
    reason: str | None = None


# Read one by one against the text. Keyed by (document number, citation span start)
# on the v2.0 text; see `build-report.md` for the reading behind each.
DECISIONS: dict[tuple[str, int], Decision] = {
    ("022", 23049): Decision(
        action="reattribute",
        to_locator=("214", "F.3d", "1058"),
        why=(
            "The sentence before it names Advanced Textile and the pin cite is 1072-73, "
            "which lies inside 214 F.3d 1058 and 127 pages below the first page of "
            "403 F. Supp. 1199. Resolution is positional and absorbed a reference "
            "eyecite never produced."
        ),
    ),
    ("006", 17868): Decision(
        action="out_of_scope",
        reason="nested_citation",
        why=(
            "`Rosenblatt v. Baer, 383 U.S. at 85` appears inside what Anaya cites, not "
            "as this filing's own citation. The filing never gives it in full."
        ),
    ),
    ("005", 12263): Decision(
        action="out_of_scope",
        reason="record_reference",
        why="`Id. ¶¶26-28` points at numbered paragraphs of a motion in this case, not a case.",
    ),
    ("022", 21683): Decision(
        action="no_authority",
        reason="authority_identified_by_docket_only",
        why=(
            "`Doe , at 3 -4` returns to Arizona Student Doe 2 v. Trump, 4:25-cv-00175 "
            "(D. Ariz.), which this document identifies only in its table of "
            "authorities and only by docket. It is a real authority and the occurrence "
            "is in scope, but this bench's authority set is reporter-identified cases, "
            "so there is nothing here for it to point at."
        ),
    ),
}

# Document 016 recites an indictment and a forfeiture complaint paragraph by
# paragraph. Every `Id.` in this range refers to those filings' numbered
# allegations. A paragraph pin cite alone would not establish this -- several
# state courts number opinion paragraphs -- but the recited subject matter does.
RECORD_REFERENCE_RANGE = ("016", 13700, 20400)


def body(path: Path) -> str:
    """The document text spans index into."""
    return path.read_text(encoding="utf-8")


def document_number(path: Path) -> str:
    """The leading number a bench document is known by."""
    return path.stem.split("__", 1)[0]


def pin_cite_written(
    text: str,
    citation: ExtractedCitation,
    limit: int,
) -> tuple[str, int, int] | None:
    """The pin cite this occurrence states, verbatim, with its span.

    Read from the text rather than from the parse. Two reasons, both learned the
    hard way: eyecite discards a pin cite it cannot parse while still making the
    attribution, so a label taken from its output cannot see its own mistakes;
    and our canonical `ReferenceCitation` carries no pin cite at all, so
    `Rafiyev at 861` would lose its page.

    ``limit`` is where the next citation begins, and reading stops there. That
    boundary is what keeps a parallel citation from being read as a page: a
    filing giving one case in two reporters writes `390 U.S. 727, 88 S.Ct.
    1323`, and to a regex over raw text the `88` is a second pin cite. It is not
    ambiguous to the extractor, which has already tokenized `88 S.Ct. 1323` as a
    citation of its own -- so the boundary is taken from there rather than
    guessed at from the characters ahead.

    A short form writes the page inside the identifier (`556 U.S. at 678`), so
    the scan starts at that `at`. Everything else states it afterwards.
    """
    start = citation.locator_span.end
    pattern = _PIN_INSIDE.get(citation.citation.kind)
    if pattern is not None:
        locator = text[citation.locator_span.start : citation.locator_span.end]
        inside = pattern.search(locator)
        if inside:
            start = citation.locator_span.start + inside.start()
    if limit <= start:
        return None
    match = PIN_FROM_TEXT.match(text, start, limit)
    if not match:
        return None
    pin = match.group("pin")
    if not pin.strip():
        return None
    return pin, match.start("pin"), match.end("pin")


def returns_to_a_statute(text: str, citation: ExtractedCitation) -> bool:
    """Whether a section sign stands where this occurrence's page would be.

    Positive evidence that the citation returns to a statute rather than to the
    case it was attributed to, and the reason the occurrence is recorded out of
    scope instead of merely having no pin cite. Absence of a pin cite means the
    document stated no page; a section sign means it stated something that is
    not a page, and the two must not be recorded the same way.

    Unlike the pin cite, this is not bounded by where the next citation begins.
    eyecite emits a token of its own for a bare `§`, so the boundary would fall
    immediately before the very character that is the evidence. Only the first
    non-space character is examined, so the look cannot reach into a following
    citation's content: `Id. See 8 U.S.C. § 1231` finds `S`, not `§`.
    """
    return bool(_SECTION_AHEAD.match(text, citation.locator_span.end))


def pin_cite_limits(citations: Sequence[ExtractedCitation], length: int) -> dict[str, int]:
    """Where each citation's pin cite must stop: the start of the next citation.

    Keyed by citation id. Citations are sorted by where their text begins, and
    each one's limit is the earliest start after it -- `min` rather than the
    next element, because eyecite nests spans (a parallel citation opens inside
    the full span of the one before it).
    """
    ordered = sorted(citations, key=lambda item: item.locator_span.start)
    limits: dict[str, int] = {}
    boundary = length
    for item in reversed(ordered):
        limits[item.citation_id] = boundary
        boundary = min(boundary, item.locator_span.start)
    return limits


def locator_of(citation: ExtractedCitation) -> tuple[str, str, str] | None:
    """The volume, reporter and page a full case citation names.

    The reporter is the canonical spelling, so that two spellings of one
    reporter key one authority. What the document wrote is kept beside it in the
    record.
    """
    inner = citation.citation
    if not isinstance(inner, FullCaseCitation):
        return None
    if not (inner.volume and inner.reporter and inner.page):
        return None
    return (inner.volume, inner.reporter.canonical, inner.page)


def decision_for(number: str, citation: ExtractedCitation) -> Decision | None:
    """The annotation that applies to this occurrence, if one was recorded."""
    keyed = DECISIONS.get((number, citation.full_span.start))
    if keyed is not None:
        return keyed
    document, low, high = RECORD_REFERENCE_RANGE
    if number == document and low <= citation.full_span.start <= high:
        return Decision(
            action="out_of_scope",
            reason="record_reference",
            why=(
                "Inside the passage reciting the indictment and forfeiture complaint, "
                "where every `Id.` points at those filings' numbered allegations."
            ),
        )
    return None


def build_document(
    path: Path,
    stated: list[dict],
) -> tuple[list[dict], list[dict], list[str]]:
    """Authorities and occurrences for one document, plus notes for the report.

    ``stated`` is the locator-only bench's ground truth for this document. It
    is inclusive of locators no tokenizer reaches, and those have to appear
    here too -- an authority a filing states is an authority whether or not
    extraction found it.
    """
    number = document_number(path)
    text = body(path)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
    tree = build_citation_tree(document)
    limits = pin_cite_limits(document.citations, len(text))
    notes: list[str] = []

    authorities: dict[str, dict] = {}
    by_locator: dict[tuple[str, str, str], str] = {}
    for authority in tree.authorities:
        root = authority.root
        locator = locator_of(root)
        if locator is None:
            notes.append(f"{number}: authority root at {root.full_span.start} states no full locator")
            continue
        authority_id = f"{number}:{root.locator_span.start}-{root.locator_span.end}"
        inner = root.citation
        authorities[authority.authority_id] = {
            "authority_id": authority_id,
            "document": path.name,
            "identifier": {
                "kind": "reporter",
                "volume": locator[0],
                "reporter": locator[1],
                "reporter_as_written": inner.reporter.as_written,
                "page": locator[2],
            },
            # Recorded, but the least reliable field here: a filing writes party
            # names as it pleases and the parse of them is a guess. Nothing in
            # this bench is scored on it.
            "case_name_as_written": {
                "plaintiff": inner.plaintiff,
                "defendant": inner.defendant,
            },
            "first_primary": {
                "span": {"start": root.locator_span.start, "end": root.locator_span.end},
                "matched_text": text[root.locator_span.start : root.locator_span.end],
            },
        }
        by_locator.setdefault(locator, authority_id)

    occurrences: list[dict] = []

    def record(
        citation: ExtractedCitation,
        *,
        authority_id: str | None,
        depth: int | None,
        scope: str,
        reason: str | None = None,
        note: str | None = None,
    ) -> None:
        pin = pin_cite_written(text, citation, limits[citation.citation_id])
        occurrences.append(
            {
                "occurrence_id": f"{number}:{citation.locator_span.start}-{citation.locator_span.end}",
                "document": path.name,
                "authority_id": authority_id,
                "kind": citation.citation.kind.value,
                "role": "first_primary" if depth == 0 else "return",
                "span": {"start": citation.full_span.start, "end": citation.full_span.end},
                "locator_span": {
                    "start": citation.locator_span.start,
                    "end": citation.locator_span.end,
                },
                "matched_text": text[citation.locator_span.start : citation.locator_span.end],
                "pin_cite_written": (
                    None if pin is None else {"text": pin[0], "span": {"start": pin[1], "end": pin[2]}}
                ),
                "scope": scope,
                **({"reason": reason} if reason else {}),
                **({"note": note} if note else {}),
            }
        )

    for authority in tree.authorities:
        record_id = authorities.get(authority.authority_id, {}).get("authority_id")
        for occurrence in authority.occurrences:
            citation = occurrence.citation
            decision = decision_for(number, citation)
            if decision is None:
                if citation.citation.kind in SECONDARY_KINDS and returns_to_a_statute(text, citation):
                    record(
                        citation,
                        authority_id=None,
                        depth=occurrence.depth,
                        scope="out_of_scope",
                        reason="statute_reference",
                        note="A section sign stands where its page would be.",
                    )
                    notes.append(
                        f"{number}: the {citation.citation.kind.value} at "
                        f"{citation.full_span.start} states a section, not a page, so it "
                        f"returns to a statute rather than to the case the tree gave it"
                    )
                    continue
                record(
                    citation,
                    authority_id=record_id,
                    depth=occurrence.depth,
                    scope="in_scope",
                )
                continue
            if decision.action == "reattribute":
                target = by_locator.get(decision.to_locator)
                if target is None:
                    notes.append(
                        f"{number}: cannot reattribute the citation at {citation.full_span.start}; "
                        f"{' '.join(decision.to_locator)} is not an authority in this document"
                    )
                    target = record_id
                record(
                    citation,
                    authority_id=target,
                    depth=occurrence.depth,
                    scope="in_scope",
                    note=decision.why,
                )
                notes.append(
                    f"{number}: the {citation.citation.kind.value} at {citation.full_span.start} is "
                    f"recorded under {' '.join(decision.to_locator)}; the tree gave it "
                    f"{authorities.get(authority.authority_id, {}).get('identifier')}. {decision.why}"
                )
            else:
                record(
                    citation,
                    authority_id=None,
                    depth=occurrence.depth,
                    scope="out_of_scope",
                    reason=decision.reason,
                    note=decision.why,
                )
                notes.append(
                    f"{number}: the {citation.citation.kind.value} at {citation.full_span.start} is "
                    f"recorded out of scope ({decision.reason}), though the tree attributed it. "
                    f"{decision.why}"
                )

    for citation in tree.unattributed:
        decision = decision_for(number, citation)
        if decision is None:
            record(
                citation,
                authority_id=None,
                depth=None,
                scope="unresolved",
                reason="no_antecedent_found",
                note="Neither attributed by the tree nor settled by reading.",
            )
            notes.append(
                f"{number}: the {citation.citation.kind.value} at {citation.full_span.start} "
                f"({text[citation.full_span.start : citation.full_span.end]!r}) is unresolved"
            )
        elif decision.action == "no_authority":
            record(
                citation,
                authority_id=None,
                depth=None,
                scope="in_scope",
                reason=decision.reason,
                note=decision.why,
            )
        else:
            record(
                citation,
                authority_id=None,
                depth=None,
                scope="out_of_scope",
                reason=decision.reason,
                note=decision.why,
            )

    for citation in tree.out_of_scope:
        if citation.citation.kind not in SECONDARY_KINDS | {CitationKind.FULL_CASE}:
            continue
        record(
            citation,
            authority_id=None,
            depth=None,
            scope="out_of_scope",
            reason="resolved_to_a_non_case",
            note="An `id.` or reference whose antecedent is a statute or unparsed span.",
        )

    reached = {
        (item["locator_span"]["start"], item["locator_span"]["end"])
        for item in occurrences
        if item["kind"] == CitationKind.FULL_CASE.value
    }
    for record in stated:
        key = (record["span"]["start"], record["span"]["end"])
        if key in reached:
            continue
        locator = (record["volume"], record["reporter"], record["page"])
        authority_id = by_locator.get(locator)
        role = "return"
        if authority_id is None:
            authority_id = f"{number}:{key[0]}-{key[1]}"
            role = "first_primary"
            by_locator[locator] = authority_id
            authorities[authority_id] = {
                "authority_id": authority_id,
                "document": path.name,
                "identifier": {
                    "kind": "reporter",
                    "volume": locator[0],
                    "reporter": locator[1],
                    "page": locator[2],
                },
                "case_name_as_written": {"plaintiff": None, "defendant": None},
                "first_primary": {
                    "span": {"start": key[0], "end": key[1]},
                    "matched_text": record["matched_text"],
                },
                "found_by_extraction": False,
                **{k: record[k] for k in ("region", "in_table", "note") if k in record},
            }
        # This locator is not one of the extracted citations, so it has no
        # entry in `limits`; its boundary is the nearest citation starting after
        # it, or the end of the text.
        boundary = min(
            (item.locator_span.start for item in document.citations if item.locator_span.start >= key[1]),
            default=len(text),
        )
        pin = PIN_FROM_TEXT.match(text, key[1], boundary)
        occurrences.append(
            {
                "occurrence_id": f"{number}:{key[0]}-{key[1]}",
                "document": path.name,
                "authority_id": authority_id,
                "kind": CitationKind.FULL_CASE.value,
                "role": role,
                "span": {"start": key[0], "end": key[1]},
                "locator_span": {"start": key[0], "end": key[1]},
                "matched_text": record["matched_text"],
                "pin_cite_written": (
                    None
                    if pin is None
                    else {
                        "text": pin.group("pin"),
                        "span": {"start": pin.start("pin"), "end": pin.end("pin")},
                    }
                ),
                "scope": "in_scope",
                "found_by_extraction": False,
                **{k: record[k] for k in ("region", "in_table", "note") if k in record},
            }
        )
        notes.append(
            f"{number}: {record['matched_text']!r} at {key[0]} is stated by the filing and "
            f"extraction does not reach it; recorded as a {role} of "
            f"{' '.join(locator)} from the locator bench"
        )

    return list(authorities.values()), occurrences, notes


def check_against_locator_bench(occurrences: list[dict], bench: Path) -> list[str]:
    """Every locator the locator-only bench states must appear as an occurrence.

    That bench is inclusive -- it holds locators no tokenizer reaches -- so this
    is where a gap in the tree bench shows up rather than being absorbed.
    """
    if not bench.exists():
        return [f"locator bench not found at {bench}; cross-check skipped"]
    stated: set[tuple[str, int, int]] = set()
    for line in bench.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        stated.add((record["document"], record["span"]["start"], record["span"]["end"]))
    found = {
        (item["document"], item["locator_span"]["start"], item["locator_span"]["end"])
        for item in occurrences
        if item["kind"] == CitationKind.FULL_CASE.value
    }
    notes = []
    for document, start, end in sorted(stated - found):
        notes.append(f"{document} [{start}:{end}] is in the locator bench and has no occurrence here")
    for document, start, end in sorted(found - stated):
        notes.append(f"{document} [{start}:{end}] is an occurrence here and not in the locator bench")
    return notes


def _refuse_to_rebuild(out: Path) -> None:
    """Stop before overwriting a bench that is now maintained directly.

    Two things make a rebuild wrong rather than merely redundant. The bench
    carries hand corrections made since it was built -- attributions read from
    the text, labels the audit changed -- and a rebuild reverts them. And its
    ids are ordinals assigned once, `008-a09` and the like, deliberately not
    derived from anything: a rebuild renumbers from scratch, which is the one
    thing an assigned-once id must never do, and every record keyed by one
    would silently point somewhere else.

    The script is kept because it is the record of how the bench was first
    built. Pass `--out` somewhere else to build a comparison copy.
    """
    if (out / "occurrences.jsonl").exists():
        msg = (
            f"{out} already holds a bench. It is maintained directly now, not rebuilt: a "
            f"rebuild reverts hand corrections and renumbers ids that are assigned once. "
            f"Pass --out to build a comparison copy somewhere else."
        )
        raise SystemExit(msg)


def main() -> int:
    """Write the tree bench and the report of what it could not settle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text",
        type=Path,
        default=Path("data/extraction-v2.0/documents_txt"),
    )
    parser.add_argument(
        "--locator-bench",
        type=Path,
        default=Path("data/extraction-v2.0/locators.jsonl"),
    )
    parser.add_argument("--out", type=Path, default=Path("data/runs/tree-rebuild"))
    args = parser.parse_args()
    _refuse_to_rebuild(args.out)

    all_authorities: list[dict] = []
    all_occurrences: list[dict] = []
    notes: list[str] = []
    stated_by_document: dict[str, list[dict]] = {}
    if args.locator_bench.exists():
        for line in args.locator_bench.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                stated_by_document.setdefault(record["document"], []).append(record)

    for path in sorted(args.text.glob("*.txt")):
        authorities, occurrences, document_notes = build_document(path, stated_by_document.get(path.name, []))
        all_authorities.extend(authorities)
        all_occurrences.extend(occurrences)
        notes.extend(document_notes)

    notes.extend(check_against_locator_bench(all_occurrences, args.locator_bench))

    args.out.mkdir(parents=True, exist_ok=True)
    text_out = args.out / "documents_txt"
    if text_out.exists():
        shutil.rmtree(text_out)
    shutil.copytree(args.text, text_out)

    with (args.out / "authorities.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_authorities:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (args.out / "occurrences.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_occurrences:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    in_scope = [item for item in all_occurrences if item["scope"] == "in_scope"]
    with_pin = [item for item in in_scope if item["pin_cite_written"]]
    secondary = [item for item in in_scope if item["kind"] != CitationKind.FULL_CASE.value]
    lines = [
        "# Build report",
        "",
        f"- authorities: {len(all_authorities)}",
        f"- occurrences: {len(all_occurrences)}",
        f"-   in scope: {len(in_scope)} ({len(secondary)} of them returns to an authority)",
        f"-   out of scope: {sum(1 for i in all_occurrences if i['scope'] == 'out_of_scope')}",
        f"-   unresolved: {sum(1 for i in all_occurrences if i['scope'] == 'unresolved')}",
        f"- occurrences stating a pin cite: {len(with_pin)}",
        "",
        "## Everything the build could not settle, or settled by hand",
        "",
    ]
    lines.extend(f"- {note}" for note in notes)
    (args.out / "build-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{len(all_authorities)} authorities, {len(all_occurrences)} occurrences")
    print(f"{len(secondary)} in-scope returns, {len(with_pin)} occurrences state a pin cite")
    print(f"{len(notes)} notes -> {args.out / 'build-report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
