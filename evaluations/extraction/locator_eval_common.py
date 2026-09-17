"""Shared corpus setup for isolated locator occurrence and colocation evaluations."""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mellea_lrc.extraction import Document, grow_roots, stable
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.serialization import deserialize_document, serialize_document

CASE_LOCATOR_KINDS = frozenset({"FullCaseCitation", "DocketCitation"})
DEFAULT_DATA = Path("/Users/woodygoodenough/CodingProjects/mellea-lrc-datasets")


@dataclass(frozen=True, slots=True)
class GrownAnnotation:
    """One annotated document and its output from ``grow_roots``."""

    document_name: str
    citation_rows: tuple[dict[str, Any], ...]
    document: Document


def grow_annotated_corpus(
    annotations: Path | None = None,
    texts_root: Path | None = None,
) -> tuple[GrownAnnotation, ...]:
    """Run stable ``grow_roots`` on every document in annotation-v4.0."""
    annotation_root = annotations or DEFAULT_DATA / "annotation-v4.0" / "documents"
    source_root = texts_root or DEFAULT_DATA
    grown: list[GrownAnnotation] = []

    for path in sorted(annotation_root.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not rows or rows[0].get("unit") != "header":
            continue
        header, citation_rows = rows[0], rows[1:]
        document_name = str(header["document"])
        text_path = source_root / Path(str(header["text"]["path"]))
        preprocessed = preprocess(text_path.read_text(encoding="utf-8"))

        # Keep extraction's diagnostics out of the JSON score and score the
        # same serialized Document boundary as the service contract.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = grow_roots(preprocessed, rules=stable())
        document = deserialize_document(serialize_document(document))
        grown.append(
            GrownAnnotation(
                document_name=document_name,
                citation_rows=tuple(citation_rows),
                document=document,
            )
        )

    return tuple(grown)


def locator_span(row: dict[str, Any]) -> tuple[int, int] | None:
    """Read an annotation's locator coordinates."""
    locator = row.get("locator")
    if not isinstance(locator, dict):
        return None
    return int(locator["start"]), int(locator["end"])


def score_sets(gold: set[Any], predicted: set[Any]) -> dict[str, int | float]:
    """Score exact set membership with precision, recall, and F1."""
    true_positive = len(gold & predicted)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gold": len(gold),
        "predicted": len(predicted),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
