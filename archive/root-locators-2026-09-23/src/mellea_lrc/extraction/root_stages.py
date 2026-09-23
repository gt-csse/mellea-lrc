"""Structural root formation for complete locator occurrences.

Root formation is deliberately separate from locator discovery and from root
identity validation.  It writes the filing-internal graph: one first occurrence
of each complete identifier is the root and later exact repetitions point at it.
It does not decide whether parallel locators name the same case; co-location is
only a boundary for field readers, and CourtListener identity remains a later
validation decision.
"""

from __future__ import annotations

from mellea_lrc.model.citations import DocketCitation, FullCaseCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.operations import assign_root
from mellea_lrc.model.record import CitationRecord, Node, Reads

ROOT_FORMATION_STAGE = "root_formation"
_MADE_BY = "mellea_lrc.extraction.root_stages"


def form_roots(document: Document) -> Document:
    """Write the root pointer for every complete locator in ``document``.

    A reporter root is the earliest occurrence of an exact ``volume reporter
    page`` identifier. A docket root is the earliest occurrence of an exact
    ``court + docket number`` identifier. Courtless dockets intentionally stay
    separate roots: later docket-only lookup must see each occurrence rather
    than inherit an unverified court from another occurrence.

    Co-location is never an input to this stage. Parallel citations remain
    distinct roots even when they share a co-location group, because only
    identity validation may establish that they reach one authority.
    """
    if ROOT_FORMATION_STAGE in document.passes:
        return document

    targets = tuple(
        record
        for record in document.active_citations
        if isinstance(record.fields, (FullCaseCitation, DocketCitation))
    )
    _require_unformed(targets)

    leaders: dict[tuple[str, ...], str] = {}
    formed: dict[str, CitationRecord] = {}
    for record in sorted(targets, key=_citation_order):
        key = _root_key(record)
        leader_id = leaders.setdefault(key, record.citation_id)
        attached = leader_id != record.citation_id
        node = Node(
            node_id=f"{ROOT_FORMATION_STAGE}:{record.citation_id}",
            reads=Reads.RECORD,
            stage=ROOT_FORMATION_STAGE,
            made_by=_MADE_BY,
            outcome="root_attached" if attached else "root_created",
            details={
                "root_id": leader_id,
                "identifier_kind": "reporter" if isinstance(record.fields, FullCaseCitation) else "docket",
            },
        )
        formed[record.citation_id] = assign_root(
            record,
            leader_id,
            node,
            resolves_to=leader_id if attached else None,
        )

    citations = tuple(formed.get(record.citation_id, record) for record in document.citations)
    return document.evolve(citations=citations, passes=(*document.passes, ROOT_FORMATION_STAGE))


def _require_unformed(records: tuple[CitationRecord, ...]) -> None:
    """Refuse to overwrite an existing graph with a different formation rule."""
    for record in records:
        if record.root_id is not None or record.resolves_to is not None:
            msg = (
                f"Cannot form roots after graph pointers exist for {record.citation_id!r}; "
                "resume the prior root-formation checkpoint or start from locator discovery."
            )
            raise ValueError(msg)


def _root_key(record: CitationRecord) -> tuple[str, ...]:
    """Return the complete, filing-stated identifier used for exact repetition."""
    citation = record.fields
    if isinstance(citation, FullCaseCitation):
        return (
            "reporter",
            _normalize(citation.volume),
            _normalize(_reporter_text(citation)),
            _normalize(citation.page),
        )
    if citation.court:
        return ("docket", _normalize(citation.court), _normalize(citation.docket_number))
    # Courtless dockets must stay independent roots. The citation id makes this
    # key intentionally unique without inventing a court relationship.
    return ("courtless_docket", record.citation_id)


def _normalize(value: str | None) -> str:
    """Compare an identifier without casing, punctuation, or whitespace noise."""
    return "".join(character for character in (value or "").casefold() if character.isalnum())


def _reporter_text(citation: FullCaseCitation) -> str | None:
    """Accept legacy string reporters while preferring the canonical spelling."""
    reporter = citation.reporter
    return reporter.as_written if hasattr(reporter, "as_written") else reporter  # type: ignore[return-value]


def _citation_order(record: CitationRecord) -> tuple[int, int, str]:
    return record.locator_span.start, record.locator_span.end, record.citation_id
