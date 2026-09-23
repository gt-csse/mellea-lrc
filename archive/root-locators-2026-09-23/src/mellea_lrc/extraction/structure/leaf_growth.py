"""Deterministic second growth of leaves over the document's current roots."""

from __future__ import annotations

from mellea_lrc.extraction.eyecite_extractor import grow_leaves as _read_and_attach_leaves
from mellea_lrc.extraction.root_stages import ROOT_FORMATION_STAGE
from mellea_lrc.extraction.structure.attachment import Attachment
from mellea_lrc.model.document import Document

LEAF_GROWTH_STAGE = "leaf_growth"


async def grow_leaves(
    document: Document,
    *,
    attach: Attachment = Attachment.STATED,
) -> Document:
    """Attach short forms to existing roots without requiring root validation.

    Site hunting and pin-cite review are independent opt-in stages. This one
    keeps the deterministic reader's leaf attachment policy and writes a
    checkpoint name distinct from its low-level eyecite `leaves` marker.
    """
    if ROOT_FORMATION_STAGE not in document.passes:
        msg = "Leaf growth requires root_formation before attachment."
        raise ValueError(msg)
    if LEAF_GROWTH_STAGE in document.passes:
        return document

    grown = _read_and_attach_leaves(document, attach=attach)
    # Keep the low-level marker: a later checkpoint must retain every pass
    # recorded by the earlier one, even when this API adds a clearer name.
    return grown.evolve(passes=(*grown.passes, LEAF_GROWTH_STAGE))
