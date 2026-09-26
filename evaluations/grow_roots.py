"""Precision of decisions made by the first two grow_roots stages."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mellea_lrc.extraction.docket_locator import find_docket_locators
from mellea_lrc.extraction.full_reporter_locator import find_full_reporter_locators
from mellea_lrc.model import Document, FullDocketCitation, FullReporterCitation

REPORTER_STAGE = "full_reporter_locators"
DOCKET_STAGE = "docket_locators"
STAGES = (REPORTER_STAGE, DOCKET_STAGE)
GOLD_KIND = {REPORTER_STAGE: "FullCaseCitation", DOCKET_STAGE: "DocketCitation"}


@dataclass(frozen=True)
class Precision:
    correct: int = 0
    total: int = 0

    def __add__(self, other: Precision) -> Precision:
        return Precision(self.correct + other.correct, self.total + other.total)

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "correct": self.correct,
            "total": self.total,
            "precision": self.correct / self.total if self.total else None,
        }


@dataclass(frozen=True)
class StagePrecision:
    stage: str
    span: Precision
    normalization: Precision

    def __add__(self, other: StagePrecision) -> StagePrecision:
        if self.stage != other.stage:
            raise ValueError("Cannot combine different stages")
        return StagePrecision(self.stage, self.span + other.span, self.normalization + other.normalization)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "span": self.span.as_dict(),
            "normalization": self.normalization.as_dict(),
        }


def _reporter_label(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _normalized_agrees(
    citation: FullReporterCitation | FullDocketCitation, identifier: dict[str, Any]
) -> bool:
    reading = citation.locator[0]
    value = reading.get_normalized()
    if isinstance(citation, FullReporterCitation):
        return (
            value.volume == int(identifier["volume"])
            and _reporter_label(value.edition) == _reporter_label(str(identifier["reporter"]))
            and value.page == str(identifier["page"])
        )
    return value.docket_number == identifier["docket_number"]


def _complete_identifier(stage: str, row: dict[str, Any]) -> dict[str, Any] | None:
    identifier = row.get("identifier")
    if not isinstance(identifier, dict):
        return None
    if stage == REPORTER_STAGE:
        if identifier.get("kind") != "reporter" or any(
            identifier.get(key) in (None, "") for key in ("volume", "reporter", "page")
        ):
            return None
        try:
            int(identifier["volume"])
        except (ValueError, TypeError):
            return None
    elif stage == DOCKET_STAGE:
        if identifier.get("kind") != "docket" or not identifier.get("docket_number"):
            return None
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    return identifier


def score_document(document: Document, rows: tuple[dict[str, Any], ...], stage: str) -> StagePrecision:
    """Score only locator fields created at one completed stage.

    A normalized output is scored only when its own exact locator span has a
    complete normalized target on that annotation row. Root links are not
    substituted: a leaf may write a different locator for the same case.
    """
    if stage not in STAGES:
        raise ValueError(f"Unsupported stage: {stage}")
    checkpoint = document.get_stage(stage)
    gold: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("kind") != GOLD_KIND[stage]:
            continue
        locator = row.get("locator")
        if not isinstance(locator, dict):
            raise ValueError(f"Annotated {GOLD_KIND[stage]} has no locator")
        key = (int(locator["start"]), int(locator["end"]))
        if key in gold:
            raise ValueError(f"Duplicate annotated {GOLD_KIND[stage]} locator: {key}")
        if checkpoint.text[key[0] : key[1]] != locator["quote"]:
            raise ValueError(f"Annotated locator differs from source: {row.get('id')}")
        gold[key] = row

    expected_type = FullReporterCitation if stage == REPORTER_STAGE else FullDocketCitation
    span = Precision()
    normalization = Precision()
    for citation in checkpoint.citations:
        if citation.nodes[0].stage != stage:
            continue
        if not isinstance(citation, expected_type):
            raise TypeError(f"Unexpected citation created by {stage}: {type(citation).__name__}")
        reading = citation.locator[0]
        if reading.node_id != citation.nodes[0].id:
            raise ValueError("Locator decision does not point to its creation node")
        key = (reading.span.start, reading.span.end)
        row = gold.get(key)
        span = Precision(span.correct + int(row is not None), span.total + 1)
        if row is None or not reading.normalizable:
            continue
        target = _complete_identifier(stage, row)
        if target is None:
            continue
        normalization = Precision(
            normalization.correct + int(_normalized_agrees(citation, target)),
            normalization.total + 1,
        )
    return StagePrecision(stage, span, normalization)


def _load_rows(
    path: Path, data_root: Path, set_name: str, filename: str, metadata: dict[str, Any]
) -> tuple[Path, tuple[dict[str, Any], ...]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty annotation file: {path}")
    header = json.loads(lines[0])
    if (
        header.get("unit") != "header"
        or header.get("document") != filename
        or header.get("dataset") != set_name
        or header.get("text", {}).get("sha256") != metadata["sha256"]
        or header.get("text", {}).get("length") != metadata["length"]
    ):
        raise ValueError(f"Annotation header does not match source: {path}")
    source_path = (data_root / header["text"]["path"]).resolve()
    if not source_path.is_relative_to((data_root / set_name).resolve()) or source_path.name != filename:
        raise ValueError(f"Annotation source path is outside this dataset: {path}")
    rows = tuple(row for line in lines[1:] if (row := json.loads(line)).get("unit") == "citation")
    return source_path, rows


def evaluate_set(data_root: Path, set_name: str, *, run_dir: Path | None = None) -> dict[str, Any]:
    """Load saved Documents, or run only the first two rule stages from source."""
    base = data_root / set_name
    manifest = json.loads((base / "documents.json").read_text(encoding="utf-8"))["documents"]
    totals = {stage: StagePrecision(stage, Precision(), Precision()) for stage in STAGES}
    for filename, metadata in sorted(manifest.items()):
        source_path, rows = _load_rows(
            base / "documents" / f"{Path(filename).stem}.jsonl", data_root, set_name, filename, metadata
        )
        if run_dir is None:
            document = Document.from_source(source_path)
            document = find_full_reporter_locators(document)
            document = find_docket_locators(document)
        else:
            artifact = run_dir / "documents" / set_name / f"{filename}.json"
            document = Document.model_validate_json(artifact.read_text(encoding="utf-8"))
        digest = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
        if digest != metadata["sha256"] or len(document.text) != metadata["length"]:
            raise ValueError(f"Document differs from dataset manifest: {set_name}/{filename}")
        for stage in STAGES:
            totals[stage] += score_document(document, rows, stage)
    return {"set": set_name, "stages": [totals[stage].as_dict() for stage in STAGES]}


def render_markdown(results: list[dict[str, Any]]) -> str:
    """Render the same two precision measures for every set and stage."""

    def cell(metric: dict[str, Any]) -> str:
        if metric["total"] == 0:
            return "—"
        return f"{metric['correct']}/{metric['total']} ({metric['precision']:.1%})"

    lines = [
        "# Grow-roots extraction precision",
        "",
        "| Set | Stage | Span precision | Normalization precision |",
        "| --- | --- | ---: | ---: |",
    ]
    for result in results:
        for stage in result["stages"]:
            lines.append(
                f"| {result['set']} | {stage['stage']} | {cell(stage['span'])} | "
                f"{cell(stage['normalization'])} |"
            )
    lines.extend(
        (
            "",
            "Span precision requires an exact same-kind annotated locator span. "
            "Normalization precision is conditional on that span and on a complete "
            "normalized identifier in the same annotation row; unnormalizable readings "
            "and rows without that target are not normalization decisions scored here.",
            "",
        )
    )
    return "\n".join(lines)
