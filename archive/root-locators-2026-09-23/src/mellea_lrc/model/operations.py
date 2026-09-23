"""Stage-neutral changes to citation records.

A node records what a reader did. These operations materialize zero or more
consequences of that reading. A stage may use one node for several field
updates, a resolution, and a judgement without inventing extra model calls.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from typing import Any

from mellea_lrc.model.citations import CanonicalCitation, CitationField, CitationKind, empty_citation
from mellea_lrc.model.document import Document
from mellea_lrc.model.record import (
    WITHDRAWN,
    WITHDRAWN_HEAD_ID,
    CitationRecord,
    CreateOperation,
    LinkOperation,
    Node,
    OperationKind,
    Question,
    Reads,
    Resolution,
)


def create_citation(
    citation_id: str,
    kind: CitationKind,
    node: Node,
) -> CitationRecord:
    """Initialize a citation's identity and kind, without reading its fields.

    The caller reads fields with :func:`update_fields` and assigns any root
    relationship separately before admitting the record to a ``Document``.
    """
    if node.reads is not Reads.DOCUMENT:
        msg = "Creating a citation requires evidence from the document"
        raise ValueError(msg)
    if citation_id == WITHDRAWN_HEAD_ID:
        raise ValueError("The virtual withdrawal head cannot be a citation")
    record = CitationRecord(
        citation_id=citation_id,
        fields=empty_citation(kind),
        created_by=node.node_id,
        trace=(node,),
        operations=(CreateOperation(OperationKind.CREATE, node.node_id, kind),),
    )
    return record


def field_values(citation: CanonicalCitation) -> dict[CitationField, Any]:
    """Read nonempty values from a parser result for the update operation."""
    return {
        CitationField(field.name): value
        for field in dataclass_fields(citation)
        if field.name != "kind" and (value := getattr(citation, field.name)) is not None
    }


def update_fields(
    citation: CitationRecord,
    node: Node,
    changes: Mapping[CitationField, Any],
    *,
    reason: str,
) -> CitationRecord:
    """Apply one reader's complete field result, including first readings."""
    citation._update_fields(node, changes, reason=reason)
    return citation


def update_field(
    citation: CitationRecord,
    node: Node,
    name: CitationField,
    value: Any,
    *,
    reason: str,
) -> CitationRecord:
    """Single-field spelling of :func:`update_fields`."""
    return update_fields(citation, node, {name: value}, reason=reason)


def withdraw_citation(citation: CitationRecord, node: Node) -> CitationRecord:
    """Append a root-link event attaching this record to the virtual head.

    Use ``withdraw_subtree`` for a formed root with leaves so their attachment
    is moved in the same document checkpoint.
    """
    citation._withdraw(node)
    return citation


def withdraw_subtree(document: Document, citation_id: str, node: Node) -> Document:
    """Atomically withdraw a root and its leaves in a new Document checkpoint.

    The previous checkpoint remains intact. Every affected record keeps its
    creation and prior root-link events, and gains a link to the virtual head.
    Passing a leaf id withdraws that leaf alone.
    """
    result = document.snapshot()
    by_id = {record.citation_id: record for record in result.citations}
    target = by_id.get(citation_id)
    if target is None:
        raise ValueError(f"Unknown citation {citation_id!r}")
    withdraw_citation(target, node)
    for leaf in result.citations:
        if leaf.citation_id == citation_id or leaf.root_id != citation_id or leaf.withdrawn:
            continue
        withdraw_citation(
            leaf,
            Node(
                node_id=f"{leaf.citation_id}:withdrawn_with_root:{node.node_id}",
                reads=Reads.RECORD,
                stage=node.stage,
                made_by=__name__,
                outcome=WITHDRAWN,
                message=f"Its root {citation_id} was withdrawn, so this leaf follows it.",
                depends_on=(node.node_id,),
            ),
        )
    return result.evolve()


def observe_citation(citation: CitationRecord, node: Node) -> CitationRecord:
    """Keep a reading that reached no durable citation-state change."""
    citation._observe(node)
    return citation


def observe_document(document: Document, node: Node) -> Document:
    """Keep a reading about the document without changing a prior snapshot."""
    if node.node_id in {seen.node_id for record in document.citations for seen in record.trace}:
        msg = f"Document-level node identifier {node.node_id!r} already belongs to a citation"
        raise ValueError(msg)
    if node.node_id in {seen.node_id for seen in document.nodes}:
        return document
    return document.evolve(nodes=(*document.nodes, node))


def judge_citation(
    citation: CitationRecord,
    node: Node,
    question: Question,
    outcome: str,
    *,
    message: str | None = None,
    type: str | None = None,
) -> CitationRecord:
    """Set one answer while retaining the node that reached it."""
    citation._judge(node, question, outcome, message=message, type=type)
    return citation


def resolve_citation(citation: CitationRecord, node: Node, resolution: Resolution) -> CitationRecord:
    """Record the retrieved authority at the identity this citation states."""
    citation._resolve(node, resolution)
    return citation


def attribute_authority(citation: CitationRecord, node: Node, authority_id: str) -> CitationRecord:
    """Point a citation at the authority established by a reading."""
    if node.reads is not Reads.RECORD:
        msg = "Authority attribution requires retrieved record evidence"
        raise ValueError(msg)
    if not authority_id:
        msg = "Authority attribution requires a nonempty authority ID"
        raise ValueError(msg)
    citation._reattribute(node, authority_id)
    return citation


def mark_extraction_reviewed(citation: CitationRecord, node: Node) -> CitationRecord:
    """Record that a model finished re-reading this citation in the filing."""
    citation._mark_extraction_reviewed_by_llm(node)
    return citation


def assign_root(
    citation: CitationRecord,
    root_id: str,
    node: Node,
    *,
    resolves_to: str | None = None,
) -> CitationRecord:
    """Materialize a filing-internal root relationship on record evidence."""
    if node.reads not in {Reads.DOCUMENT, Reads.RECORD}:
        msg = "Root assignment requires document or record evidence"
        raise ValueError(msg)
    if not root_id:
        msg = "A root attachment requires a root citation ID"
        raise ValueError(msg)
    if root_id == WITHDRAWN_HEAD_ID:
        raise ValueError("Use withdraw_citation to attach the virtual withdrawal head")
    if resolves_to == citation.citation_id:
        msg = "A citation cannot resolve to itself"
        raise ValueError(msg)
    changes = []
    if citation.root_id != root_id:
        changes.append(LinkOperation(OperationKind.ROOT_LINK, node.node_id, root_id))
    if citation.resolves_to != resolves_to:
        changes.append(LinkOperation(OperationKind.ANTECEDENT_LINK, node.node_id, resolves_to))
    return replace(
        citation,
        root_id=root_id,
        root_link_node_id=node.node_id if citation.root_id != root_id else citation.root_link_node_id,
        resolves_to=resolves_to,
        operations=(*citation.operations, *changes),
        trace=_trace_with(citation, node),
    )


def assign_antecedent(citation: CitationRecord, antecedent_id: str, node: Node) -> CitationRecord:
    """Record the earlier citation this occurrence refers to, without guessing its root."""
    if not antecedent_id or antecedent_id == citation.citation_id:
        msg = "An antecedent must be another citation ID"
        raise ValueError(msg)
    changes = (
        (LinkOperation(OperationKind.ANTECEDENT_LINK, node.node_id, antecedent_id),)
        if citation.resolves_to != antecedent_id
        else ()
    )
    return replace(
        citation,
        resolves_to=antecedent_id,
        operations=(*citation.operations, *changes),
        trace=_trace_with(citation, node),
    )


def record_colocation(citation: CitationRecord, colocation_id: str | None, node: Node) -> CitationRecord:
    """Write the rule-derived locator grouping, including removal on a rerun."""
    if node.reads is not Reads.RECORD:
        msg = "Colocation assignment reads admitted locator records"
        raise ValueError(msg)
    changes = (
        (LinkOperation(OperationKind.COLOCATION_LINK, node.node_id, colocation_id),)
        if citation.colocation_id != colocation_id
        else ()
    )
    return replace(
        citation,
        colocation_id=colocation_id,
        operations=(*citation.operations, *changes),
        trace=_trace_with(citation, node),
    )


def _trace_with(citation: CitationRecord, node: Node) -> tuple[Node, ...]:
    """Include the evidence in the same immutable replacement as its effect."""
    if node.node_id in {seen.node_id for seen in citation.trace}:
        return citation.trace
    return (*citation.trace, node)
