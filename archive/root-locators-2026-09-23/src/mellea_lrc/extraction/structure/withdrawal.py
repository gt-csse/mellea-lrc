"""Move rejected citations into the virtual withdrawal tree.

The virtual head has no citation record or source span. Its root pointer is
itself, and every withdrawn citation points at it. A withdrawn root's leaves
move with it in the same document transaction. Previous attachments remain in
each record's ordered ROOT_LINK operations, so a later checkpoint contains the
whole path from creation to withdrawal.
"""

from __future__ import annotations

from collections.abc import Sequence

from mellea_lrc.model.operations import withdraw_citation
from mellea_lrc.model.record import WITHDRAWN, CitationRecord, Node, Reads

MADE_BY = "mellea_lrc.extraction.structure.withdrawal"


def withdraw_leaves_of_withdrawn_roots(
    citations: Sequence[CitationRecord], *, stage: str = "extraction"
) -> int:
    """Attach leaves grown against already-withdrawn roots to the virtual head.

    This is a repair within a stage that has just grown leaves. Normal root
    withdrawal should use :func:`mellea_lrc.model.operations.withdraw_subtree`
    so a caller never receives
    a checkpoint with active leaves under a withdrawn root.
    """
    withdrawn: dict[str, CitationRecord] = {
        record.citation_id: record for record in citations if record.withdrawn
    }
    followed = 0
    for leaf in citations:
        root = withdrawn.get(leaf.root_id or "")
        if root is None or leaf.citation_id == root.citation_id or leaf.withdrawn:
            continue
        _follow_root(leaf, root, stage=stage)
        followed += 1
    return followed


def _follow_root(leaf: CitationRecord, root: CitationRecord, *, stage: str) -> None:
    withdraw_citation(
        leaf,
        Node(
            node_id=f"{leaf.citation_id}:withdrawn_with_root:{root.root_link_node_id}",
            reads=Reads.RECORD,
            stage=stage,
            made_by=MADE_BY,
            outcome=WITHDRAWN,
            message=(
                f"The root this citation was built from, {root.citation_id}, was withdrawn, "
                "so this citation joins the virtual withdrawal head."
            ),
            depends_on=(root.root_link_node_id,) if root.root_link_node_id else (),
        ),
    )
