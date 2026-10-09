"""Load source-checked annotation rows and align citation occurrences."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mellea_lrc.model import Document
from mellea_lrc.model.citations import (
    Citation,
    FullDocketCitation,
    FullReporterCitation,
    IdCitation,
    ReferenceCitation,
    ShortReporterCitation,
    SupraCitation,
)


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


def annotation_span(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, dict):
        return None
    value = value.get("source", value)
    return (value["start"], value["end"]) if "start" in value else None


def annotation_kind(citation: Citation) -> str:
    return {
        FullReporterCitation: "FullCaseCitation",
        FullDocketCitation: "DocketCitation",
        ShortReporterCitation: "ShortCaseCitation",
        IdCitation: "IdCitation",
        ReferenceCitation: "ReferenceCitation",
        SupraCitation: "SupraCitation",
    }[type(citation)]


def annotation_site(row: dict[str, Any]) -> tuple[str, int, int]:
    field = (
        "case_name"
        if row["kind"] == "ReferenceCitation"
        else (
            "locator"
            if row["kind"] in {"FullCaseCitation", "DocketCitation", "ShortCaseCitation"}
            else "cited_as"
        )
    )
    span = annotation_span(row.get(field))
    if span is None:
        raise ValueError(f"{row['id']}: missing citation source span")
    return (row["kind"], *span)


def citation_site(citation: Citation) -> tuple[str, int, int]:
    return (annotation_kind(citation), citation.site_span.start, citation.site_span.end)


def annotations_by_site(document: Document) -> dict[tuple[str, int, int], dict[str, Any]]:
    rows = citation_annotations(document)
    for row in rows:
        if row["kind"] == "ReferenceCitation" and annotation_span(row.get("pin_cite")) is None:
            raise ValueError(f"{row['id']}: reference without a pinpoint must be out_of_scope_citation")
    keyed = {annotation_site(row): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError("Duplicate gold citation site")
    return keyed


def align_citation_annotations(
    citations: Sequence[Citation], gold: dict[tuple[str, int, int], dict[str, Any]]
) -> list[tuple[Citation, dict[str, Any] | None]]:
    """Align sites once, allowing unique overlap only for name-only references.

    Exact field spans are scored independently by each workflow scorer.
    """
    result = []
    used: set[str] = set()
    for citation in citations:
        row = gold.get(citation_site(citation))
        if row is None and isinstance(citation, ReferenceCitation):
            matches = [
                candidate
                for (kind, start, end), candidate in gold.items()
                if kind == "ReferenceCitation"
                and start < citation.site_span.end
                and citation.site_span.start < end
            ]
            if len(matches) > 1:
                raise ValueError("Reference site overlaps multiple gold occurrences")
            row = matches[0] if matches else None
        if row is not None:
            if row["id"] in used:
                row = None
            else:
                used.add(row["id"])
        result.append((citation, row))
    return result
