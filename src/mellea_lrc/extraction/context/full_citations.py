"""Extraction-substage guards and contextual text readings."""

from __future__ import annotations

from mellea_lrc.model.document import Document


def require_structure(document: Document) -> None:
    """Require the pipeline position where contextual fields can be read."""
    if "grow_roots.read_root_fields.resolve_colocations" not in document.substage_runs:
        raise ValueError("Resolve colocations before reading contextual fields")
    if "grow_roots.form_root_groups.form_roots" in document.substage_runs:
        raise ValueError("Read contextual fields before forming roots")
