"""Named extraction passes and the dependencies that constrain their order.

The locator read produces spans and searches backward for case names. Those
spans then define colocation groups. A field name reader fills names on full
locators admitted after the raw parse. An independent docket audit checks for
an explicit court or a colocated reporter and withdraws unsupported candidates.
Court and date readers run after the audit,
so a parallel citation may read through its co-located neighbors and stop at
the next unrelated locator. Pin cites are structured after the date reader
finalizes each citation's full span. Root/reference assignment runs last.

There is no cycle in this order. The old docket reader created one in practice
by refusing to emit a docket until it had already resolved a court; locator
recognition now accepts a docket-shaped span with court=None, and court
resolution follows grouping. Eyecite still bundles some metadata reads inside
get_citations, so the stable profile re-reads court/date within the explicit
boundaries. That re-read is what exposes the stages without changing eyecite's
baseline when no project rules are supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

from mellea_lrc.core.citations import DocketCitation
from mellea_lrc.extraction.reading.case_names import reread_case_names
from mellea_lrc.extraction.reading.docket_audit import audit_docket_citations
from mellea_lrc.extraction.reading.pin_cite_spans import read_pin_cites
from mellea_lrc.extraction.reading.post_citation import reread_courts, reread_dates
from mellea_lrc.extraction.structure.citation_tree import assign_roots
from mellea_lrc.extraction.structure.colocation import assign_colocation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mellea_lrc.extraction.rules import ExtractionRules
    from mellea_lrc.extraction.types import CitationRecord, Document


class Pass(Protocol):
    """One refinement over a document's citations, given the text they index."""

    def __call__(self, text: str, citations: Sequence[CitationRecord]) -> tuple[CitationRecord, ...]: ...


CASE_NAME_STAGE = "case_name_resolution"
COURT_STAGE = "court_resolution"
DATE_STAGE = "date_resolution"
PIN_CITE_STAGE = "pin_cite_resolution"


def _colocation(text: str, citations: Sequence[CitationRecord]) -> tuple[CitationRecord, ...]:
    """Group citations occupying the same span and written with nothing between."""
    return assign_colocation(text, citations)


def _courts(text: str, citations: Sequence[CitationRecord]) -> tuple[CitationRecord, ...]:
    """Resolve explicit courts after locator spans and groups are known."""
    return reread_courts(text, citations)


def _case_names(text: str, citations: Sequence[CitationRecord]) -> tuple[CitationRecord, ...]:
    """Fill only absent full-locator names after co-location is known."""
    return reread_case_names(text, citations)


def _dates(text: str, citations: Sequence[CitationRecord]) -> tuple[CitationRecord, ...]:
    """Resolve dates after locator spans and groups are known."""
    return reread_dates(text, citations)


def _pin_cites(text: str, citations: Sequence[CitationRecord]) -> tuple[CitationRecord, ...]:
    """Locate and structure pin cites against the settled citation spans."""
    return read_pin_cites(text, citations)


def _root(text: str, citations: Sequence[CitationRecord]) -> tuple[CitationRecord, ...]:
    """Write onto each citation the root it refers to. Does not read the text."""
    del text
    # Courtless docket locators remain independent candidates. Dockets with an
    # explicit, resolved court can share an extraction root when the complete
    # docket identifier repeats in this document.
    leaders: dict[tuple[str, str], str] = {}
    linked: list[CitationRecord] = []
    for citation in citations:
        stated = citation.stated
        if citation.withdrawn or not isinstance(stated, DocketCitation) or not stated.court:
            linked.append(citation)
            continue
        key = (stated.court, _normalize_docket(stated.docket_number))
        leader = leaders.setdefault(key, citation.citation_id)
        linked.append(replace(citation, resolves_to=leader) if leader != citation.citation_id else citation)
    return assign_roots(linked)


def _normalize_docket(value: str | None) -> str:
    """Compare docket spellings without punctuation or whitespace changes."""
    return "".join(char for char in (value or "").casefold() if char.isalnum())


@dataclass(frozen=True, slots=True)
class Stage:
    """One pass, named, with the reason it runs where it does."""

    name: str
    run: Pass
    why: str
    """Why this position. Read as a constraint, not a description."""


STAGES: tuple[Stage, ...] = (
    Stage(
        name="colocation",
        run=_colocation,
        why=(
            "Decided from the spans eyecite produced, before anything alters them, and "
            "before court and date resolution, whose search boundaries use its ids."
        ),
    ),
    Stage(
        name="case_names",
        run=_case_names,
        why=(
            "Runs after co-location so a newly admitted full locator reads the name before "
            "the first identifier of its site; existing raw names remain untouched."
        ),
    ),
    Stage(
        name="docket_audit",
        run=audit_docket_citations,
        why=(
            "Needs colocation to retain dockets supported by a reporter. Checks the court "
            "rule independently before any court metadata is written."
        ),
    ),
    Stage(
        name="courts",
        run=_courts,
        why=(
            "Runs after locators and co-location so court search can use the group span; "
            "the audit has already decided admission, and this pass scans again to write courts."
        ),
    ),
    Stage(
        name="dates",
        run=_dates,
        why=(
            "Needs co-location ids: a citation may read past a co-located neighbour for "
            "the date at the end, and must stop at any other locator. It trims the full "
            "citation span, so it runs after locator and name reading."
        ),
    ),
    Stage(
        name="pin_cites",
        run=_pin_cites,
        why=(
            "Runs after dates because the bounded date reader finalizes each full citation span; "
            "pin-cite parsing locates eyecite's page string inside that settled span."
        ),
    ),
    Stage(
        name="root",
        run=_root,
        why=(
            "Last, and only because there is no reason to be earlier: it reads "
            "`resolves_to`, which nothing before it touches. Running it after the spans "
            "and dates are settled keeps the citations it writes onto final."
        ),
    ),
)


def refine(
    text: str,
    citations: Sequence[CitationRecord],
    rules: ExtractionRules | None = None,
) -> tuple[CitationRecord, ...]:
    """Run configured readers in dependency order, then assign citation roots.

    With no rules, locator parsing and metadata come from eyecite. Project
    readers are opt-in and supplied one stage at a time through ExtractionRules.
    """
    refined = tuple(citations)
    if rules is not None and rules.colocation_reader is not None:
        refined = rules.colocation_reader(text, refined)
    if rules is not None and rules.case_name_field_reader is not None:
        refined = rules.case_name_field_reader(text, refined)
    if rules is not None and rules.docket_auditor is not None:
        refined = rules.docket_auditor(text, refined)
    if rules is not None and rules.court_reader is not None:
        refined = rules.court_reader(text, refined)
    if rules is not None and rules.date_reader is not None:
        refined = rules.date_reader(text, refined)
    if rules is not None and rules.pin_cite_reader is not None:
        refined = rules.pin_cite_reader(text, refined)
    else:
        refined = _pin_cites(text, refined)
    return _root(text, refined)


def audit_dockets(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Audit already-grouped docket candidates, preserving raw locator records.

    Admission uses an independent court scan or a colocated reporter. Court
    fields are left for resolve_courts. With no rules, this stage is a no-op.
    """
    if rules is None or rules.docket_auditor is None:
        return document
    audited = replace(document, citations=rules.docket_auditor(document.text, document.citations))
    # The public stage can also be replayed after leaf growth.
    from mellea_lrc.extraction.structure.withdrawal import withdraw_leaves_of_withdrawn_roots

    withdraw_leaves_of_withdrawn_roots(audited)
    return audited


def resolve_case_names(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Fill names absent from full locators after colocation is known.

    Eyecite's raw parse already supplied most names during locator extraction.
    This explicit pass is for a full locator added later, such as a
    site-admitted docket, and never overwrites a name already recorded.
    """
    if CASE_NAME_STAGE in document.passes:
        return document
    citations = (
        rules.case_name_field_reader(document.text, document.citations)
        if rules is not None and rules.case_name_field_reader is not None
        else document.citations
    )
    return replace(document, citations=citations, passes=(*document.passes, CASE_NAME_STAGE))


def resolve_courts(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Resolve court context after locator spans and colocation groups exist."""
    if COURT_STAGE in document.passes:
        return document
    citations = (
        rules.court_reader(document.text, document.citations)
        if rules is not None and rules.court_reader is not None
        else document.citations
    )
    return replace(document, citations=citations, passes=(*document.passes, COURT_STAGE))


def resolve_dates(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Resolve decision dates after locator spans and colocation groups exist."""
    if DATE_STAGE in document.passes:
        return document
    citations = (
        rules.date_reader(document.text, document.citations)
        if rules is not None and rules.date_reader is not None
        else document.citations
    )
    return replace(document, citations=citations, passes=(*document.passes, DATE_STAGE))


def resolve_pin_cites(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Structure eyecite's pin-cite text against finalized citation spans."""
    if PIN_CITE_STAGE in document.passes:
        return document
    reader = rules.pin_cite_reader if rules is not None else None
    if reader is None:
        reader = read_pin_cites
    return replace(
        document,
        citations=reader(document.text, document.citations),
        passes=(*document.passes, PIN_CITE_STAGE),
    )
