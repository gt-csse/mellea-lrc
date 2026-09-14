"""A withdrawn root takes its leaves with it, and neither is deleted.

Withdrawing says the document does not hold this citation: a statute read as a
case, a docket number that is a record entry, a root that reaches no authority.
The record stays and stays addressable -- `root_id` and `authority_id` name
citation ids, and a record that goes away takes every reference to it along.

A leaf is built from a root and means nothing without one, so a leaf whose root
is withdrawn is withdrawn too. **It keeps its `root_id`.** The pointer is what
says which root took it, it is what a later pass follows if that root is ever
admitted again, and a leaf with the pointer cleared would be a leaf standing on
nothing -- which is the one thing `CitationRecord` refuses to be.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.core.record import WITHDRAWN, Node, Reads

if TYPE_CHECKING:
    from mellea_lrc.core.record import CitationRecord
    from mellea_lrc.extraction.types import Document

#: What makes the node: it reads the document's own records rather than the
#: filing's text, so it corrects nothing and settles nothing. It is the reason
#: one record followed another out.
MADE_BY = "mellea_lrc.extraction.structure.withdrawal"


def withdraw_leaves_of_withdrawn_roots(document: Document, *, stage: str = "extraction") -> int:
    """Withdraw every leaf whose root is withdrawn. Returns how many followed.

    Run after anything withdraws a root, and after the leaves are grown: both
    orders reach the same document, which is why there is one function rather
    than a rule in each caller. Running it again withdraws nothing, because a
    leaf already withdrawn is not withdrawn twice.
    """
    withdrawn: dict[str, CitationRecord] = {
        record.citation_id: record for record in document.citations if record.withdrawn
    }
    if not withdrawn:
        return 0
    followed = 0
    for record in document.citations:
        root = withdrawn.get(record.root_id or "")
        if root is None or record.withdrawn or record.citation_id == root.citation_id:
            continue
        record.withdraw(
            Node(
                node_id=f"{record.citation_id}:withdrawn_with_root",
                reads=Reads.RECORD,
                stage=stage,
                made_by=MADE_BY,
                outcome=WITHDRAWN,
                message=(
                    f"The root this citation is built from, {root.citation_id}, was withdrawn, "
                    "so there is no citation here to be right or wrong about."
                ),
                depends_on=(root.withdrawn_by,) if root.withdrawn_by else (),
            )
        )
        followed += 1
    return followed
