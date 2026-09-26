"""Score individual grow_roots stages and final root fields from one Document."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mellea_lrc.model import Document, FullDocketCitation, FullReporterCitation
from mellea_lrc.model.citations.full import FullCitation
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID, latest

REPORTER_STAGE = "full_reporter_locators"
DOCKET_STAGE = "docket_locators"
HUNT_STAGE = "docket_locator_site_hunting"
ENTRY_STAGE = "docket_entries"
COLOCATION_STAGE = "colocations"
CASE_NAME_STAGE = "case_names"
COURT_STAGE = "courts"
DATE_STAGE = "dates"
PIN_STAGE = "pin_cites"
ROOT_STAGE = "roots"


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
class FieldScore:
    correct: int = 0
    predicted: int = 0
    gold: int = 0

    def __add__(self, other: FieldScore) -> FieldScore:
        return FieldScore(
            self.correct + other.correct, self.predicted + other.predicted, self.gold + other.gold
        )

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "correct": self.correct,
            "predicted": self.predicted,
            "gold": self.gold,
            "precision": self.correct / self.predicted if self.predicted else None,
            "recall": self.correct / self.gold if self.gold else None,
        }


@dataclass(frozen=True)
class StageScore:
    stage: str
    metrics: dict[str, Precision]

    def __add__(self, other: StageScore) -> StageScore:
        if self.stage != other.stage or self.metrics.keys() != other.metrics.keys():
            raise ValueError("Cannot combine different stage scores")
        return StageScore(
            self.stage, {key: value + other.metrics[key] for key, value in self.metrics.items()}
        )

    def as_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "metrics": {key: value.as_dict() for key, value in self.metrics.items()}}


@dataclass(frozen=True)
class WorkflowScore:
    stages: tuple[StageScore, ...]
    root_fields: dict[str, dict[str, FieldScore]]

    def __add__(self, other: WorkflowScore) -> WorkflowScore:
        if tuple(item.stage for item in self.stages) != tuple(item.stage for item in other.stages):
            raise ValueError("Cannot combine workflows with different stage runs")
        return WorkflowScore(
            tuple(left + right for left, right in zip(self.stages, other.stages, strict=True)),
            {
                name: {
                    measure: value + other.root_fields[name][measure] for measure, value in measures.items()
                }
                for name, measures in self.root_fields.items()
            },
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "stages": [item.as_dict() for item in self.stages],
            "root_fields": {
                name: {measure: score.as_dict() for measure, score in measures.items()}
                for name, measures in self.root_fields.items()
            },
        }


def _rows(document: Document) -> tuple[dict[str, Any], ...]:
    """Resolve official annotations from the source path; reject stale gold."""
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
    expected_path = (dataset.parent / text.get("path", "")).resolve()
    if (
        header.get("unit") != "header"
        or header.get("dataset") != dataset.name
        or header.get("document") != source.name
        or expected_path != source
        or text.get("length") != len(document.text)
        or text.get("sha256") != hashlib.sha256(document.text.encode("utf-8")).hexdigest()
    ):
        raise ValueError(f"Annotation does not match Document source: {annotation}")
    rows = tuple(row for line in lines[1:] if (row := json.loads(line)).get("unit") == "citation")
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        if row.get("kind") not in {"FullCaseCitation", "DocketCitation"}:
            continue
        locator = row.get("locator")
        if not isinstance(locator, dict):
            raise ValueError(f"Full citation has no locator: {row.get('id')}")
        key = (row["kind"], locator["start"], locator["end"])
        if key in seen:
            raise ValueError(f"Duplicate annotated locator: {key}")
        seen.add(key)
        for name in ("locator", "case_name", "court", "date", "pin_cite", "docket_entry"):
            target = row.get(name)
            if isinstance(target, dict) and "start" in target:
                if document.text[target["start"] : target["end"]] != target["quote"]:
                    raise ValueError(f"Annotated {name} differs from source: {row.get('id')}")
    return rows


def _kind(citation: FullCitation) -> str:
    return "FullCaseCitation" if isinstance(citation, FullReporterCitation) else "DocketCitation"


def _key(citation: FullCitation) -> tuple[str, int, int]:
    span = citation.locator[-1].span
    return _kind(citation), span.start, span.end


def _gold(rows: tuple[dict[str, Any], ...]) -> dict[tuple[str, int, int], dict[str, Any]]:
    return {
        (row["kind"], row["locator"]["start"], row["locator"]["end"]): row
        for row in rows
        if row.get("kind") in {"FullCaseCitation", "DocketCitation"}
    }


def _span(reading: Any) -> tuple[int, int] | None:
    return None if reading is None or reading.span is None else (reading.span.start, reading.span.end)


def _gold_span(target: Any) -> tuple[int, int] | None:
    return (target["start"], target["end"]) if isinstance(target, dict) and "start" in target else None


def score_full_reporter_locators(document: Document) -> StageScore:
    checkpoint = document.get_stage(REPORTER_STAGE)
    gold = _gold(_rows(checkpoint))
    span_correct = span_total = norm_correct = norm_total = 0
    for citation in checkpoint.full_locators:
        if not isinstance(citation, FullReporterCitation):
            continue
        row = gold.get(_key(citation))
        node_ids = {node.id for node in citation.nodes if node.stage == REPORTER_STAGE}
        for reading in citation.locator:
            if reading.node_id not in node_ids:
                continue
            if reading.span is not None:
                span_total += 1
                span_correct += int(_span(reading) == _gold_span(row.get("locator") if row else None))
            if not reading.normalizable:
                continue
            if row is None:
                norm_total += 1
                continue
            identifier = row.get("identifier")
            if (
                not isinstance(identifier, dict)
                or identifier.get("kind") != "reporter"
                or any(identifier.get(key) in (None, "") for key in ("volume", "reporter", "page"))
            ):
                continue
            value = reading.get_normalized()
            norm_total += 1
            norm_correct += int(
                str(value.volume) == str(identifier["volume"])
                and re.sub(r"\s+", "", value.edition).casefold()
                == re.sub(r"\s+", "", str(identifier["reporter"])).casefold()
                and value.page == str(identifier["page"])
            )
    return StageScore(
        REPORTER_STAGE,
        {"span": Precision(span_correct, span_total), "normalization": Precision(norm_correct, norm_total)},
    )


def score_docket_locators(document: Document) -> StageScore:
    checkpoint = document.get_stage(DOCKET_STAGE)
    gold = _gold(_rows(checkpoint))
    span_correct = span_total = norm_correct = norm_total = 0
    for citation in checkpoint.full_locators:
        if not isinstance(citation, FullDocketCitation):
            continue
        row = gold.get(_key(citation))
        node_ids = {node.id for node in citation.nodes if node.stage == DOCKET_STAGE}
        for reading in citation.locator:
            if reading.node_id not in node_ids:
                continue
            if reading.span is not None:
                span_total += 1
                span_correct += int(_span(reading) == _gold_span(row.get("locator") if row else None))
            if not reading.normalizable:
                continue
            if row is None:
                norm_total += 1
                continue
            identifier = row.get("identifier")
            if (
                not isinstance(identifier, dict)
                or identifier.get("kind") != "docket"
                or not identifier.get("docket_number")
            ):
                continue
            norm_total += 1
            norm_correct += int(reading.get_normalized().docket_number == identifier["docket_number"])
    return StageScore(
        DOCKET_STAGE,
        {"span": Precision(span_correct, span_total), "normalization": Precision(norm_correct, norm_total)},
    )


def score_docket_locator_site_hunting(document: Document) -> StageScore:
    checkpoint = document.get_stage(HUNT_STAGE)
    gold = _gold(_rows(checkpoint))
    span_correct = span_total = norm_correct = norm_total = 0
    for citation in checkpoint.full_locators:
        if not isinstance(citation, FullDocketCitation):
            continue
        row = gold.get(_key(citation))
        node_ids = {node.id for node in citation.nodes if node.stage == HUNT_STAGE}
        for reading in citation.locator:
            if reading.node_id not in node_ids:
                continue
            if reading.span is not None:
                span_total += 1
                span_correct += int(_span(reading) == _gold_span(row.get("locator") if row else None))
            if not reading.normalizable:
                continue
            if row is None:
                norm_total += 1
                continue
            identifier = row.get("identifier")
            if (
                not isinstance(identifier, dict)
                or identifier.get("kind") != "docket"
                or not identifier.get("docket_number")
            ):
                continue
            norm_total += 1
            norm_correct += int(reading.get_normalized().docket_number == identifier["docket_number"])
    return StageScore(
        HUNT_STAGE,
        {"span": Precision(span_correct, span_total), "normalization": Precision(norm_correct, norm_total)},
    )


def score_docket_entries(document: Document) -> StageScore:
    checkpoint = document.get_stage(ENTRY_STAGE)
    gold = _gold(_rows(checkpoint))
    span_correct = span_total = norm_correct = norm_total = 0
    for citation in checkpoint.full_locators:
        if not isinstance(citation, FullDocketCitation):
            continue
        row = gold.get(_key(citation))
        node_ids = {node.id for node in citation.nodes if node.stage == ENTRY_STAGE}
        for reading in citation.docket_entry:
            if reading.node_id not in node_ids:
                continue
            target = row.get("docket_entry") if row else None
            if reading.span is not None:
                span_total += 1
                span_correct += int(_span(reading) == _gold_span(target))
            if not reading.normalizable:
                continue
            if row is None:
                norm_total += 1
                continue
            if isinstance(target, dict) and target.get("number") is None:
                continue
            norm_total += 1
            norm_correct += int(
                isinstance(target, dict) and str(reading.get_normalized()) == str(target["number"])
            )
    return StageScore(
        ENTRY_STAGE,
        {"span": Precision(span_correct, span_total), "normalization": Precision(norm_correct, norm_total)},
    )


def score_case_names(document: Document) -> StageScore:
    checkpoint = document.get_stage(CASE_NAME_STAGE)
    gold = _gold(_rows(checkpoint))
    span_correct = span_total = norm_correct = norm_total = 0
    for citation in checkpoint.full_locators:
        row = gold.get(_key(citation))
        node_ids = {node.id for node in citation.nodes if node.stage == CASE_NAME_STAGE}
        for reading in citation.case_name:
            if reading.node_id not in node_ids:
                continue
            target = row.get("case_name") if row else None
            if reading.span is not None:
                span_total += 1
                span_correct += int(_span(reading) == _gold_span(target))
            if not reading.normalizable:
                continue
            if row is None:
                norm_total += 1
                continue
            if isinstance(target, dict) and target.get("normalized") is None:
                continue
            norm_total += 1
            norm_correct += int(
                isinstance(target, dict)
                and reading.get_normalized().model_dump(mode="json") == target["normalized"]
            )
    return StageScore(
        CASE_NAME_STAGE,
        {"span": Precision(span_correct, span_total), "normalization": Precision(norm_correct, norm_total)},
    )


def score_courts(document: Document) -> StageScore:
    checkpoint = document.get_stage(COURT_STAGE)
    gold = _gold(_rows(checkpoint))
    span_correct = span_total = norm_correct = norm_total = 0
    for citation in checkpoint.full_locators:
        row = gold.get(_key(citation))
        node_ids = {node.id for node in citation.nodes if node.stage == COURT_STAGE}
        for reading in citation.court:
            if reading.node_id not in node_ids:
                continue
            target = row.get("court") if row else None
            if reading.span is not None:
                span_total += 1
                span_correct += int(_span(reading) == _gold_span(target))
            if not reading.normalizable:
                continue
            if row is None:
                norm_total += 1
                continue
            if isinstance(target, dict) and target.get("id") is None:
                continue
            norm_total += 1
            norm_correct += int(isinstance(target, dict) and reading.get_normalized().id == target["id"])
    return StageScore(
        COURT_STAGE,
        {"span": Precision(span_correct, span_total), "normalization": Precision(norm_correct, norm_total)},
    )


def score_dates(document: Document) -> StageScore:
    checkpoint = document.get_stage(DATE_STAGE)
    gold = _gold(_rows(checkpoint))
    span_correct = span_total = norm_correct = norm_total = 0
    for citation in checkpoint.full_locators:
        row = gold.get(_key(citation))
        node_ids = {node.id for node in citation.nodes if node.stage == DATE_STAGE}
        for reading in citation.date:
            if reading.node_id not in node_ids:
                continue
            target = row.get("date") if row else None
            if reading.span is not None:
                span_total += 1
                span_correct += int(_span(reading) == _gold_span(target))
            if not reading.normalizable:
                continue
            if row is None:
                norm_total += 1
                continue
            if isinstance(target, dict) and target.get("normalized") is None:
                continue
            norm_total += 1
            if not isinstance(target, dict):
                continue
            value = reading.get_normalized()
            actual = f"{value.year:04d}"
            if value.month is not None:
                actual += f"-{value.month:02d}"
            if value.day is not None:
                actual += f"-{value.day:02d}"
            norm_correct += int(actual == target["normalized"])
    return StageScore(
        DATE_STAGE,
        {"span": Precision(span_correct, span_total), "normalization": Precision(norm_correct, norm_total)},
    )


def score_pin_cites(document: Document) -> StageScore:
    checkpoint = document.get_stage(PIN_STAGE)
    gold = _gold(_rows(checkpoint))
    span_correct = span_total = norm_correct = norm_total = 0
    for citation in checkpoint.full_locators:
        row = gold.get(_key(citation))
        node_ids = {node.id for node in citation.nodes if node.stage == PIN_STAGE}
        for reading in citation.pin_cite:
            if reading.node_id not in node_ids:
                continue
            target = row.get("pin_cite") if row else None
            if reading.span is not None:
                span_total += 1
                span_correct += int(_span(reading) == _gold_span(target))
            if not reading.normalizable:
                continue
            if row is None:
                norm_total += 1
                continue
            if isinstance(target, dict) and target.get("normalized") is None:
                continue
            norm_total += 1
            norm_correct += int(
                isinstance(target, dict)
                and [item.model_dump(mode="json", exclude_none=True) for item in reading.get_normalized()]
                == target["normalized"]
            )
    return StageScore(
        PIN_STAGE,
        {"span": Precision(span_correct, span_total), "normalization": Precision(norm_correct, norm_total)},
    )


def score_colocations(document: Document) -> StageScore:
    checkpoint = document.get_stage(COLOCATION_STAGE)
    gold = _gold(_rows(checkpoint))
    by_id = {citation.id: citation for citation in checkpoint.full_locators}
    available = {_key(citation) for citation in checkpoint.full_locators}
    groups: dict[str, set[str]] = {}
    for citation in checkpoint.full_locators:
        group_id = latest(citation.colocation_id)
        if group_id is not None:
            groups.setdefault(group_id, set()).add(citation.id)
    correct = 0
    for ids in groups.values():
        predicted = {_key(by_id[citation_id]) for citation_id in ids}
        rows = [gold.get(key) for key in predicted]
        if not rows or any(row is None or row.get("colocation_id") is None for row in rows):
            continue
        gold_ids = {row["colocation_id"] for row in rows}
        if len(gold_ids) == 1:
            gold_available = {
                key for key, row in gold.items() if key in available and row.get("colocation_id") in gold_ids
            }
            correct += int(predicted == gold_available)
    return StageScore(COLOCATION_STAGE, {"groups": Precision(correct, len(groups))})


def score_roots(document: Document) -> StageScore:
    checkpoint = document.get_stage(ROOT_STAGE)
    gold = _gold(_rows(checkpoint))
    by_id = {citation.id: citation for citation in checkpoint.full_locators}
    available_gold_ids = {
        row["id"] for citation in checkpoint.full_locators if (row := gold.get(_key(citation))) is not None
    }
    correct = total = 0
    for citation in checkpoint.full_locators:
        if not any(node.stage == ROOT_STAGE for node in citation.nodes):
            continue
        root_id = latest(citation.root_id)
        if root_id is None or root_id == WITHDRAWN_ROOT_ID:
            continue
        total += 1
        row = gold.get(_key(citation))
        root = by_id.get(root_id)
        root_row = gold.get(_key(root)) if root is not None else None
        if row is None or root_row is None:
            continue
        # A missed locator must not become a root-formation error. If the
        # annotated representative was found, however, the chosen head matters.
        if row["root_id"] in available_gold_ids:
            correct += int(root_row["id"] == row["root_id"])
        else:
            correct += int(root_row["root_id"] == row["root_id"])
    return StageScore(ROOT_STAGE, {"root_assignment": Precision(correct, total)})


GROW_ROOTS_STAGES: tuple[tuple[str, Callable[[Document], StageScore]], ...] = (
    (REPORTER_STAGE, score_full_reporter_locators),
    (DOCKET_STAGE, score_docket_locators),
    (HUNT_STAGE, score_docket_locator_site_hunting),
    (ENTRY_STAGE, score_docket_entries),
    (COLOCATION_STAGE, score_colocations),
    (CASE_NAME_STAGE, score_case_names),
    (COURT_STAGE, score_courts),
    (DATE_STAGE, score_dates),
    (PIN_STAGE, score_pin_cites),
    (ROOT_STAGE, score_roots),
)


def _root_has_normalized_gold(name: str, row: dict[str, Any]) -> bool:
    """Availability of gold for the final-root summary only."""
    if name in {"full_reporter_locator", "docket_locator"}:
        identifier = row.get("identifier")
        if not isinstance(identifier, dict):
            return False
        if name == "docket_locator":
            return identifier.get("kind") == "docket" and bool(identifier.get("docket_number"))
        return identifier.get("kind") == "reporter" and all(
            identifier.get(key) not in (None, "") for key in ("volume", "reporter", "page")
        )
    target = row.get(name)
    if not isinstance(target, dict):
        return False
    normalized_key = {
        "docket_entry": "number",
        "case_name": "normalized",
        "court": "id",
        "date": "normalized",
        "pin_cite": "normalized",
    }[name]
    return target.get(normalized_key) is not None


def _root_normalization_unannotated(name: str, row: dict[str, Any]) -> bool:
    if name in {"full_reporter_locator", "docket_locator"}:
        return not _root_has_normalized_gold(name, row)
    return isinstance(row.get(name), dict) and not _root_has_normalized_gold(name, row)


def _root_normalization_agrees(name: str, reading: Any, row: dict[str, Any]) -> bool:
    value = reading.get_normalized()
    if name == "full_reporter_locator":
        target = row["identifier"]
        return (
            str(value.volume) == str(target["volume"])
            and re.sub(r"\s+", "", value.edition).casefold()
            == re.sub(r"\s+", "", str(target["reporter"])).casefold()
            and value.page == str(target["page"])
        )
    if name == "docket_locator":
        return value.docket_number == row["identifier"]["docket_number"]
    if name == "docket_entry":
        return str(value) == str(row["docket_entry"]["number"])
    if name == "case_name":
        return value.model_dump(mode="json") == row["case_name"]["normalized"]
    if name == "court":
        return value.id == row["court"]["id"]
    if name == "date":
        actual = f"{value.year:04d}"
        if value.month is not None:
            actual += f"-{value.month:02d}"
        if value.day is not None:
            actual += f"-{value.day:02d}"
        return actual == row["date"]["normalized"]
    if name == "pin_cite":
        return [item.model_dump(mode="json", exclude_none=True) for item in value] == row["pin_cite"][
            "normalized"
        ]
    raise ValueError(f"Unknown root field: {name}")


def score_grow_roots_workflow(document: Document) -> WorkflowScore:
    """All stage precision, then precision/recall for final root fields."""
    final = document.get_stage(ROOT_STAGE)
    missing = [
        stage for stage, _ in GROW_ROOTS_STAGES if stage != HUNT_STAGE and stage not in final.stage_runs
    ]
    if missing:
        raise ValueError(f"Incomplete grow_roots workflow; missing stages: {', '.join(missing)}")
    stages = tuple(
        score(final)
        for stage, score in GROW_ROOTS_STAGES
        if stage != HUNT_STAGE or HUNT_STAGE in final.stage_runs
    )
    gold = _gold(tuple(row for row in _rows(final) if row.get("is_root")))
    names = (
        "full_reporter_locator",
        "docket_locator",
        "docket_entry",
        "case_name",
        "court",
        "date",
        "pin_cite",
    )
    attributes = {
        "full_reporter_locator": "locator",
        "docket_locator": "locator",
        "docket_entry": "docket_entry",
        "case_name": "case_name",
        "court": "court",
        "date": "date",
        "pin_cite": "pin_cite",
    }
    result: dict[str, dict[str, FieldScore]] = {}
    for name in names:
        gold_kind = (
            "FullCaseCitation"
            if name == "full_reporter_locator"
            else "DocketCitation"
            if name in {"docket_locator", "docket_entry"}
            else None
        )
        gold_rows = [row for row in gold.values() if gold_kind is None or row["kind"] == gold_kind]
        target_name = "locator" if name.endswith("_locator") else name
        span_gold = sum(_gold_span(row.get(target_name)) is not None for row in gold_rows)
        norm_gold = sum(_root_has_normalized_gold(name, row) for row in gold_rows)
        span_correct = span_predicted = norm_correct = norm_predicted = 0
        for citation in final.roots:
            if gold_kind is not None and _kind(citation) != gold_kind:
                continue
            if name == "docket_entry" and not isinstance(citation, FullDocketCitation):
                continue
            readings = getattr(citation, attributes[name])
            if not readings:
                continue
            reading = readings[-1]
            row = gold.get(_key(citation))
            target = row.get(target_name) if row else None
            span_match = reading.span is not None and _span(reading) == _gold_span(target)
            if reading.span is not None:
                span_predicted += 1
                span_correct += int(span_match)
            if reading.normalizable and not (row is not None and _root_normalization_unannotated(name, row)):
                norm_predicted += 1
                norm_correct += int(
                    row is not None
                    and _root_has_normalized_gold(name, row)
                    and _root_normalization_agrees(name, reading, row)
                )
        result[name] = {
            "span": FieldScore(span_correct, span_predicted, span_gold),
            "normalization": FieldScore(norm_correct, norm_predicted, norm_gold),
        }
    return WorkflowScore(stages, result)


def render_markdown(result: dict[str, Any]) -> str:
    """Render only the stage and root-field measures in the score."""

    def precision(value: dict[str, Any]) -> str:
        return (
            "—" if value["total"] == 0 else f"{value['correct']}/{value['total']} ({value['precision']:.1%})"
        )

    def field(value: dict[str, Any], key: str) -> str:
        denominator = value["predicted" if key == "precision" else "gold"]
        return "—" if denominator == 0 else f"{value['correct']}/{denominator} ({value[key]:.1%})"

    lines = [
        f"# Grow-roots evaluation: {result['set']}",
        "",
        "Docket site hunting: "
        + ("included" if any(stage["stage"] == HUNT_STAGE for stage in result["stages"]) else "not run"),
        "",
        "## Stage precision",
        "",
        "| Stage | Decision | Precision |",
        "| --- | --- | ---: |",
    ]
    for stage in result["stages"]:
        for name, value in stage["metrics"].items():
            lines.append(f"| {stage['stage']} | {name} | {precision(value)} |")
    lines += [
        "",
        "## Root fields",
        "",
        "| Field | Span precision | Span recall | Normalization precision | Normalization recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, measures in result["root_fields"].items():
        span, norm = measures["span"], measures["normalization"]
        lines.append(
            f"| {name} | {field(span, 'precision')} | {field(span, 'recall')} | "
            f"{field(norm, 'precision')} | {field(norm, 'recall')} |"
        )
    lines.append("")
    return "\n".join(lines)
