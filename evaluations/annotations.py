"""Load official annotation files and verify their source-text provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mellea_lrc.model import Document


def load_annotations(document: Document) -> tuple[dict[str, Any], ...]:
    """Read all native rows after checking the header against the Document.

    Workflow scorers own row eligibility and required gold-field validation.
    This loader checks only the shared file and source-text contract.
    """
    if document.source_path is None:
        raise ValueError("Evaluation needs a Document with an official source path")
    source = Path(document.source_path).resolve()
    dataset = source.parent.parent
    annotation = dataset / "documents" / f"{source.stem}.jsonl"
    lines = annotation.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty annotation file: {annotation}")
    header = json.loads(lines[0])
    text = header.get("text", {})
    if (
        header.get("unit") != "header"
        or header.get("dataset") != dataset.name
        or header.get("document") != source.name
        or (dataset.parent / text.get("path", "")).resolve() != source
        or text.get("length") != len(document.text)
        or text.get("sha256") != hashlib.sha256(document.text.encode("utf-8")).hexdigest()
    ):
        raise ValueError(f"Official annotation does not match Document source: {annotation}")
    return tuple(json.loads(line) for line in lines[1:])


def citation_annotations(document: Document) -> tuple[dict[str, Any], ...]:
    """Read rows explicitly included by the native citation annotation unit."""
    return tuple(row for row in load_annotations(document) if row.get("unit") == "citation")
