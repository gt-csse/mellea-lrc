"""Read annotation rows only after loading a saved prediction artifact."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mellea_lrc.model import Document, Span

SETS = (
    "primary",
    "hallucination-set-1",
    "hallucination-set-2",
    "reliable-high-profile",
    "reliable-low-profile",
)
SPAN_FIELDS = ("locator", "case_name", "court", "date", "pin_cite", "docket_entry")


def span(raw: dict[str, Any]) -> Span:
    return Span(start=int(raw["start"]), end=int(raw["end"]))


def annotated_documents(
    data_root: Path, run_dir: Path, sets: tuple[str, ...]
) -> Iterator[tuple[str, str, Document, tuple[dict[str, Any], ...]]]:
    """Load each manifest-listed artifact and its matching source annotation.

    A missing document, unfinished stage, or changed source must fail loudly;
    silently skipping it would turn an incomplete run into a better score.
    """
    if not sets or any(name not in SETS for name in sets):
        raise ValueError("Select one or more known annotated sets")
    for name in dict.fromkeys(sets):
        manifest_path = data_root / name / "documents.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["documents"]
        for filename, metadata in sorted(manifest.items()):
            artifact_path = run_dir / "documents" / name / f"{filename}.json"
            document = Document.model_validate_json(artifact_path.read_text(encoding="utf-8"))
            digest = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
            if digest != metadata["sha256"] or len(document.text) != metadata["length"]:
                raise ValueError(f"{artifact_path}: source differs from documents.json")
            if document.index_spans != tuple(span(raw) for raw in metadata.get("index_spans", ())):
                raise ValueError(f"{artifact_path}: index masks differ from documents.json")
            annotation_path = data_root / name / "documents" / f"{Path(filename).stem}.jsonl"
            lines = annotation_path.read_text(encoding="utf-8").splitlines()
            if not lines:
                raise ValueError(f"{annotation_path}: empty annotation file")
            header = json.loads(lines[0])
            if (
                header.get("unit") != "header"
                or header.get("document") != filename
                or header.get("text", {}).get("sha256") != digest
            ):
                raise ValueError(f"{annotation_path}: annotation header differs from the saved document")
            rows = []
            for line in lines[1:]:
                row = json.loads(line)
                if row.get("unit") != "citation":
                    continue
                locator = row.get("locator")
                if isinstance(locator, dict):
                    for field in SPAN_FIELDS:
                        evidence = row.get(field)
                        if not isinstance(evidence, dict) or "start" not in evidence:
                            continue
                        evidence_span = span(evidence)
                        if document.text[evidence_span.start : evidence_span.end] != evidence.get("quote"):
                            raise ValueError(
                                f"{annotation_path}: {row.get('id')} {field} quote differs from source"
                            )
                    rows.append(row)
            yield name, filename, document, tuple(rows)
