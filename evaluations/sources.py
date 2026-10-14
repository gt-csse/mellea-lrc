"""Native evaluation sources bound to their official frozen TXT rendering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mellea_lrc.model import Document, PreprocessedDocument, Rule
from mellea_lrc.model.citations.tags import CitationTagKind
from mellea_lrc.model.span import is_within


def load_source(path: Path) -> Document:
    """Bind recorded rendering metadata to the exact official text coordinates."""
    records = json.loads((path.parent.parent / "documents.json").read_text(encoding="utf-8"))["documents"]
    if not isinstance(records, dict) or path.name not in records:
        raise ValueError(f"Dataset has no rendering metadata record for source: {path}")
    rendering = records[path.name]
    if not isinstance(rendering, dict):
        raise ValueError(f"Source rendering must be a metadata record: {path}")
    raw = path.read_bytes()
    if rendering.get("sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError(f"Source rendering SHA-256 differs from the official text: {path}")
    text = raw.decode("utf-8")
    if type(rendering.get("length")) is not int or rendering["length"] != len(text):
        raise ValueError(f"Source rendering length differs from the official text: {path}")
    if rendering.get("backend") != "docling" or "backend_version" not in rendering:
        raise ValueError(f"Source rendering needs recorded Docling backend metadata: {path}")
    version = rendering["backend_version"]
    if version is not None and (not isinstance(version, str) or not version.strip()):
        raise ValueError(f"Source rendering has an invalid backend version: {path}")
    layout_rules = rendering.get("layout_rules")
    removal_rules = {Rule.MARGIN_LINE_NUMBERS.value, Rule.REPEATED_FURNITURE.value, Rule.DOCKET_STAMP.value}
    if (
        not isinstance(layout_rules, list)
        or any(not isinstance(rule, str) or rule not in removal_rules for rule in layout_rules)
        or len(layout_rules) != len(set(layout_rules))
        or type(rendering.get("table_structure")) is not bool
    ):
        raise ValueError(f"Source rendering has invalid recorded layout rules: {path}")
    rules = list(layout_rules)
    if not rendering["table_structure"]:
        rules.append(Rule.TABLE_AS_TEXT.value)
    spans = rendering.get("index_spans")
    if not isinstance(spans, list) or any(
        not isinstance(span, dict)
        or not {"start", "end"}.issubset(span)
        or set(span).difference({"start", "end", "kind"})
        or type(span["start"]) is not int
        or type(span["end"]) is not int
        for span in spans
    ):
        raise ValueError(f"Source rendering needs explicit valid table-of-authorities spans: {path}")
    rules.append(Rule.TABLE_OF_AUTHORITIES.value)
    preprocessed = PreprocessedDocument.model_validate(
        {
            "source_metadata": {"path": str(path.resolve()), "format": "text"},
            "text": text,
            "preprocessing_metadata": {"backend": "docling", "backend_version": version, "rules": rules},
            "index_spans": spans,
        }
    )
    return Document.from_preprocessed(preprocessed)


def validate_source(document: Document, *, source: Document | None = None) -> None:
    """Reject source drift or missing creation tags without mutating histories."""
    if document.source_path is None:
        raise ValueError("Saved Document has no official source path")
    source = load_source(Path(document.source_path)) if source is None else source
    if (
        source.source_path is None
        or Path(document.source_path).resolve() != Path(source.source_path).resolve()
        or document.source_metadata.format != source.source_metadata.format
    ):
        raise ValueError(f"Saved Document source differs from current source: {source.source_path}")
    for field in ("text", "preprocessing_metadata", "index_spans"):
        if getattr(document, field) != getattr(source, field):
            raise ValueError(f"Saved Document {field} differs from current source: {source.source_path}")
    for citation in document.citations:
        creation_span = citation._through_node_count(1).site_span
        expected = {
            index
            for index, component in enumerate(source.index_spans)
            if is_within(creation_span, (component,))
        }
        actual = {
            tag.component_index for tag in citation.tags if tag.kind is CitationTagKind.TABLE_OF_AUTHORITIES
        }
        if actual != expected:
            raise ValueError(
                f"Saved citation {citation.id} has missing source table-of-authorities tags; "
                "regenerate from native preprocessing"
            )
