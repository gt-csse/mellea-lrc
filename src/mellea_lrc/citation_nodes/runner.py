"""Small internal runner for citation-node operations."""

from __future__ import annotations

from typing import Protocol

from mellea_lrc.citation_nodes.types import CitationNode, CitationNodeDocument


class CitationNodeOperation(Protocol):
    """A pure-ish operation that advances one citation node."""

    name: str

    def run(self, node: CitationNode) -> CitationNode:
        """Return an updated node with any trace steps appended."""


def run_node_operation(
    document: CitationNodeDocument,
    citation_id: str,
    operation: CitationNodeOperation,
) -> CitationNodeDocument:
    """Run one operation against one citation node in a document projection."""
    changed = False
    nodes: list[CitationNode] = []
    for node in document.nodes:
        if node.citation_id != citation_id:
            nodes.append(node)
            continue
        updated = operation.run(node)
        if updated.citation_id != citation_id:
            msg = (
                f"Operation {operation.name!r} changed citation id "
                f"from {citation_id!r} to {updated.citation_id!r}"
            )
            raise ValueError(msg)
        nodes.append(updated)
        changed = True
    if not changed:
        msg = f"Unknown citation node id: {citation_id!r}"
        raise KeyError(msg)
    return CitationNodeDocument(text=document.text, nodes=tuple(nodes))


def run_operation_for_each_node(
    document: CitationNodeDocument,
    operation: CitationNodeOperation,
) -> CitationNodeDocument:
    """Run one operation independently for every citation node."""
    return CitationNodeDocument(
        text=document.text,
        nodes=tuple(operation.run(node) for node in document.nodes),
    )
