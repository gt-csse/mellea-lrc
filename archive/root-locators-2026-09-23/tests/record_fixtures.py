"""Build test citations through the same durable operations as production."""

from __future__ import annotations

from mellea_lrc.model.citations import CanonicalCitation, CitationField, citation_kind
from mellea_lrc.model.operations import (
    assign_antecedent,
    assign_root,
    create_citation,
    field_values,
    record_colocation,
    update_fields,
)
from mellea_lrc.model.record import CitationRecord, Node, Reads


def read_citation(
    citation_id: str,
    fields: CanonicalCitation,
    *,
    root_id: str | None = None,
    resolves_to: str | None = None,
) -> CitationRecord:
    """Create a citation, read its fields, and record any filing links."""
    created = Node(
        node_id=f"{citation_id}:fixture:create",
        reads=Reads.DOCUMENT,
        stage="fixture",
        made_by="test",
        outcome="created",
    )
    record = create_citation(citation_id, citation_kind(fields), created)
    if root_id is not None:
        linked = Node(
            node_id=f"{citation_id}:fixture:link",
            reads=Reads.DOCUMENT,
            stage="fixture",
            made_by="test",
            outcome="linked",
        )
        record = assign_root(record, root_id, linked, resolves_to=resolves_to)
    elif resolves_to is not None:
        linked = Node(
            node_id=f"{citation_id}:fixture:link",
            reads=Reads.DOCUMENT,
            stage="fixture",
            made_by="test",
            outcome="linked",
        )
        record = assign_antecedent(record, resolves_to, linked)
    read = Node(
        node_id=f"{citation_id}:fixture:read",
        reads=Reads.DOCUMENT,
        stage="fixture",
        made_by="test",
        outcome="read",
    )
    return update_fields(record, read, field_values(fields), reason="fixture reading")


def revise_citation(record: CitationRecord, **changes: object) -> CitationRecord:
    """Record a later fixture reading instead of mutating projected fields."""
    node = Node(
        node_id=f"{record.citation_id}:fixture:revision:{len(record.trace)}",
        reads=Reads.DOCUMENT,
        stage="fixture",
        made_by="test",
        outcome="revised",
    )
    return update_fields(
        record,
        node,
        {CitationField(name): value for name, value in changes.items()},
        reason="fixture revision",
    )


def colocate_citation(record: CitationRecord, colocation_id: str) -> CitationRecord:
    """Attach a locator group through a recorded rule reading."""
    node = Node(
        node_id=f"{record.citation_id}:fixture:colocation:{len(record.trace)}",
        reads=Reads.RECORD,
        stage="fixture",
        made_by="test",
        outcome="colocated",
    )
    return record_colocation(record, colocation_id, node)
