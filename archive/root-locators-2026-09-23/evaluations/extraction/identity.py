"""Resume a saved root-identity document for leaf evaluation.

The identity artifact is already a complete native Document checkpoint. Leaf
evaluation must use that saved state as a whole; copying selected fields onto a
fresh extraction would discard the operation history and its evidence nodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.model.document import Document

if TYPE_CHECKING:
    from pathlib import Path


def settled(document: Document, artifact: Path) -> Document:
    """Return the corresponding root-identity checkpoint when one exists."""
    if not artifact.exists():
        return document
    saved = Document.model_validate_json(artifact.read_text(encoding="utf-8"))
    if saved.text != document.text:
        raise ValueError(f"{artifact}: identity checkpoint belongs to different document text")
    return saved
