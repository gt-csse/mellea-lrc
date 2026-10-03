"""Score individual grow_roots stages and final root fields from one Document."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from evaluations.annotations import citation_annotations
from evaluations.score_types import FieldScore, Precision, StageScore
from mellea_lrc.model import Document, FullDocketCitation, FullReporterCitation
from mellea_lrc.model.citations.full import FullCitation
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID, latest
from mellea_lrc.validation.docket_root_lookup_govinfo_llm_review import STAGE as VALIDATED_FIELDS_STAGE

REPORTER_STAGE = "1_full_reporter_locators"
DOCKET_STAGE = "2_docket_locators"
HUNT_STAGE = "3_docket_locator_site_hunting"
ENTRY_STAGE = "4_docket_entries"
COLOCATION_STAGE = "5_colocations"
CASE_NAME_STAGE = "6_case_names"
COURT_STAGE = "7_courts"
DATE_STAGE = "8_dates"
PIN_STAGE = "9_pin_cites"
ROOT_STAGE = "10_roots"
ROOT_REVIEW_STAGE = "11_docket_root_llm_reassignment"
ROOT_FIELDS = ("locator", "case_name", "court", "date", "pin_cite", "docket_entry")


@dataclass(frozen=True)
class RecallScore:
    """A gold-side measure with no implied prediction-side precision."""

    correct: int = 0
    gold: int = 0

    def __add__(self, other: RecallScore) -> RecallScore:
        return RecallScore(self.correct + other.correct, self.gold + other.gold)

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "correct": self.correct,
            "gold": self.gold,
            "recall": self.correct / self.gold if self.gold else None,
        }


@dataclass(frozen=True)
class WorkflowScore:
    stages: tuple[StageScore, ...]
    root_fields: dict[str, dict[str, FieldScore | RecallScore]]
    validated_root_fields: dict[str, dict[str, FieldScore | RecallScore]] | None = None

    def __add__(self, other: WorkflowScore) -> WorkflowScore:
        if tuple(item.stage for item in self.stages) != tuple(item.stage for item in other.stages):
            raise ValueError("Cannot combine workflows with different stage runs")
        if (self.validated_root_fields is None) != (other.validated_root_fields is None):
            raise ValueError("Cannot combine workflows with different validation checkpoints")
        return WorkflowScore(
            tuple(left + right for left, right in zip(self.stages, other.stages, strict=True)),
            {
                name: {
                    measure: value + other.root_fields[name][measure] for measure, value in measures.items()
                }
                for name, measures in self.root_fields.items()
            },
            (
                {
                    name: {
                        measure: value + other.validated_root_fields[name][measure]
                        for measure, value in measures.items()
                    }
                    for name, measures in self.validated_root_fields.items()
                }
                if self.validated_root_fields is not None and other.validated_root_fields is not None
                else None
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        result = {
            "stages": [item.as_dict() for item in self.stages],
            "root_fields": {
                name: {measure: score.as_dict() for measure, score in measures.items()}
                for name, measures in self.root_fields.items()
            },
        }
        if self.validated_root_fields is not None:
            result["validated_root_fields"] = {
                name: {measure: score.as_dict() for measure, score in measures.items()}
                for name, measures in self.validated_root_fields.items()
            }
        return result


def _rows(document: Document) -> tuple[dict[str, Any], ...]:
    """Resolve official annotations from the source path; reject stale gold."""
    rows = citation_annotations(document)
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        if row.get("kind") not in {"FullCaseCitation", "DocketCitation"}:
            continue
        if row.get("is_root"):
            for name in ROOT_FIELDS:
                field = row.get(name)
                if (
                    not isinstance(field, dict)
                    or not isinstance(field.get("source"), dict)
                    or not isinstance(field.get("normalization"), dict)
                ):
                    raise ValueError(f"{row.get('id')}: missing explicit {name} gold outcome")
        locator = row.get("locator")
        if not isinstance(locator, dict):
            raise ValueError(f"Full citation has no locator: {row.get('id')}")
        locator_span = _gold_span(locator)
        if locator_span is None:
            raise ValueError(f"Full citation has no quoted locator: {row.get('id')}")
        key = (row["kind"], *locator_span)
        if key in seen:
            raise ValueError(f"Duplicate annotated locator: {key}")
        seen.add(key)
    return rows


def _kind(citation: FullCitation) -> str:
    return "FullCaseCitation" if isinstance(citation, FullReporterCitation) else "DocketCitation"


def _key(citation: FullCitation) -> tuple[str, int, int]:
    span = citation.locator[-1].span
    return _kind(citation), span.start, span.end


def _gold(rows: tuple[dict[str, Any], ...]) -> dict[tuple[str, int, int], dict[str, Any]]:
    found: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("kind") not in {"FullCaseCitation", "DocketCitation"}:
            continue
        span = _gold_span(row["locator"])
        if span is None:
            raise ValueError(f"{row.get('id')}: full citation has no locator span")
        found[(row["kind"], *span)] = row
    return found


def _span(reading: Any) -> tuple[int, int] | None:
    return None if reading is None or reading.span is None else (reading.span.start, reading.span.end)


def _gold_span(target: Any) -> tuple[int, int] | None:
    if not isinstance(target, dict):
        return None
    source = target.get("source", target)
    return (source["start"], source["end"]) if source.get("kind") == "quoted" or "start" in source else None


def _target(row: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if row is None:
        return None
    return row.get(name)


def _source_agrees(reading: Any, target: dict[str, Any] | None) -> bool:
    if target is not None and "source" in target:
        source = target["source"]
        if source["kind"] == "quoted":
            return reading is not None and _span(reading) == _gold_span(target)
        if source["kind"] == "inferred":
            return reading is not None and reading.span is None
        if source["kind"] == "not_stated":
            return reading is None
        if source["kind"] == "not_applicable":
            raise ValueError("Not-applicable fields must be excluded before scoring")
        raise AssertionError("Unknown gold source type")
    return reading is not None and _span(reading) == _gold_span(target)


def _spans_overlap(left: tuple[int, int] | None, right: tuple[int, int] | None) -> bool:
    """Require at least one shared source character in half-open spans."""
    return left is not None and right is not None and left[0] < right[1] and right[0] < left[1]


def _source_overlaps(reading: Any, target: dict[str, Any]) -> bool:
    """Relax quoted boundaries only; preserve exact absence/inference rules."""
    if target["source"]["kind"] == "quoted":
        return _spans_overlap(_span(reading), _gold_span(target))
    return _source_agrees(reading, target)


def _root_field_overlap_recall(
    roots: tuple[FullCitation, ...],
    gold_rows: list[dict[str, Any]],
    *,
    attribute: str,
    target_name: str,
) -> RecallScore:
    """Credit each predicted root to at most one overlapping gold root."""
    options: list[list[int]] = []
    for row in gold_rows:
        locator_span = _gold_span(row["locator"])
        target = row[target_name]
        candidates: list[int] = []
        for index, citation in enumerate(roots):
            if _kind(citation) != row["kind"] or not _spans_overlap(
                _span(citation.locator[-1]), locator_span
            ):
                continue
            readings = getattr(citation, attribute)
            reading = readings[-1] if readings else None
            if _source_overlaps(reading, target):
                candidates.append(index)
        options.append(candidates)

    assigned: dict[int, int] = {}

    def claim(gold_index: int, visited: set[int]) -> bool:
        for prediction_index in options[gold_index]:
            if prediction_index in visited:
                continue
            visited.add(prediction_index)
            previous = assigned.get(prediction_index)
            if previous is None or claim(previous, visited):
                assigned[prediction_index] = gold_index
                return True
        return False

    correct = sum(claim(index, set()) for index in range(len(gold_rows)))
    return RecallScore(correct, len(gold_rows))


def _nonroot_normalization_annotated(name: str, row: dict[str, Any]) -> bool:
    """Check normalized targets in the current nonroot field formats."""
    if name == "locator":
        # Full locator annotations require their own source/normalization
        # pair; a separate citation identifier is not a field target.
        return False
    target = row.get(name)
    if not isinstance(target, dict):
        return False
    key = {"docket_entry": "number", "court": "id"}.get(name, "normalized")
    value = target.get(key)
    return value is not None and (name != "pin_cite" or bool(value))


def _normalization_agrees(name: str, reading: Any, row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    target = _target(row, name)
    if not isinstance(target, dict):
        raise ValueError(f"{row['id']}: missing explicit {name} normalization gold")
    source = target.get("source", target)
    if not isinstance(source, dict):
        raise ValueError(f"{row['id']}: missing explicit {name} source gold outcome")
    source_kind = source.get("kind", "quoted" if "start" in source else None)
    if source_kind not in {"quoted", "inferred", "not_stated"}:
        raise ValueError(f"{row['id']}: missing explicit {name} source gold outcome")
    if "normalization" in target:
        normalization = target["normalization"]
        if not isinstance(normalization, dict):
            raise ValueError(f"{row['id']}: missing explicit {name} normalization gold")
        if normalization.get("kind") == "unavailable":
            if source_kind == "not_stated":
                return reading is None
            # An unavailable value is a property of this exact quoted field,
            # not permission to score any failed parser reading as correct.
            return (
                source_kind == "quoted"
                and reading is not None
                and not reading.normalizable
                and _source_agrees(reading, target)
            )
        if normalization.get("kind") != "value":
            raise ValueError(f"{row['id']}: missing explicit {name} normalization gold value")
        if normalization.get("value") is None:
            raise ValueError(f"{row['id']}: missing explicit {name} normalization gold value")
        if source_kind == "not_stated":
            raise ValueError(f"{row['id']}: {name} not_stated source requires unavailable normalization")
        value = normalization["value"]
    else:
        if source_kind == "not_stated":
            raise ValueError(f"{row['id']}: {name} not_stated source requires unavailable normalization")
        if not _nonroot_normalization_annotated(name, row):
            raise ValueError(f"{row['id']}: missing explicit {name} normalization gold")
        value = target
        if name == "case_name" or name == "pin_cite":
            value = target["normalized"]
        elif name == "date":
            value = {"normalized": target["normalized"]}
        elif name == "court":
            value = {"id": target["id"]}
        elif name == "docket_entry":
            value = target["number"]
    if reading is None or not reading.normalizable:
        return False
    normalized = reading.get_normalized()
    if name == "locator" and value["kind"] == "reporter":
        return (
            str(normalized.volume) == str(value["volume"])
            and re.sub(r"\s+", "", normalized.edition).casefold()
            == re.sub(r"\s+", "", str(value["reporter"])).casefold()
            and normalized.page == str(value["page"])
        )
    if name == "locator" and value["kind"] == "docket":
        return normalized.docket_number == value["docket_number"]
    if name == "case_name":
        return normalized.model_dump(mode="json") == value
    if name == "court":
        return normalized.id == value["id"]
    if name == "date":
        actual = f"{normalized.year:04d}"
        if normalized.month is not None:
            actual += f"-{normalized.month:02d}"
        if normalized.day is not None:
            actual += f"-{normalized.day:02d}"
        return actual == value["normalized"]
    if name == "pin_cite":
        return [item.model_dump(mode="json", exclude_none=True) for item in normalized] == value
    if name == "docket_entry":
        return str(normalized) == value
    raise ValueError(f"Unknown root field: {name}")


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
                span_correct += int(_source_agrees(reading, _target(row, "locator")))
            norm_total += 1
            norm_correct += int(_normalization_agrees("locator", reading, row))
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
                span_correct += int(_source_agrees(reading, _target(row, "locator")))
            norm_total += 1
            norm_correct += int(_normalization_agrees("locator", reading, row))
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
                span_correct += int(_source_agrees(reading, _target(row, "locator")))
            norm_total += 1
            norm_correct += int(_normalization_agrees("locator", reading, row))
    return StageScore(
        HUNT_STAGE,
        {"span": Precision(span_correct, span_total), "normalization": Precision(norm_correct, norm_total)},
    )


def score_docket_entries(document: Document) -> StageScore:
    """Score every docket's entry outcome, including absence."""
    checkpoint = document.get_stage(ENTRY_STAGE)
    gold = _gold(_rows(checkpoint))
    span_correct = span_total = norm_correct = norm_total = 0
    for citation in checkpoint.full_locators:
        if not isinstance(citation, FullDocketCitation):
            continue
        row = gold.get(_key(citation))
        reading = citation.docket_entry[-1] if citation.docket_entry else None
        span_total += 1
        span_correct += int(row is not None and _source_agrees(reading, _target(row, "docket_entry")))
        norm_total += 1
        norm_correct += int(_normalization_agrees("docket_entry", reading, row))
    return StageScore(
        ENTRY_STAGE,
        {"span": Precision(span_correct, span_total), "normalization": Precision(norm_correct, norm_total)},
    )


def score_case_names(document: Document) -> StageScore:
    """Score every full occurrence's name outcome, including absence."""
    checkpoint = document.get_stage(CASE_NAME_STAGE)
    gold = _gold(_rows(checkpoint))
    span_correct = span_total = norm_correct = norm_total = 0
    for citation in checkpoint.full_locators:
        row = gold.get(_key(citation))
        reading = citation.case_name[-1] if citation.case_name else None
        span_total += 1
        span_correct += int(row is not None and _source_agrees(reading, _target(row, "case_name")))
        norm_total += 1
        norm_correct += int(_normalization_agrees("case_name", reading, row))
    return StageScore(
        CASE_NAME_STAGE,
        {"span": Precision(span_correct, span_total), "normalization": Precision(norm_correct, norm_total)},
    )


def score_courts(document: Document) -> StageScore:
    """Score every full occurrence's court outcome, including inference and absence."""
    checkpoint = document.get_stage(COURT_STAGE)
    gold = _gold(_rows(checkpoint))
    span_correct = span_total = norm_correct = norm_total = 0
    for citation in checkpoint.full_locators:
        row = gold.get(_key(citation))
        reading = citation.court[-1] if citation.court else None
        span_total += 1
        span_correct += int(row is not None and _source_agrees(reading, _target(row, "court")))
        norm_total += 1
        norm_correct += int(_normalization_agrees("court", reading, row))
    return StageScore(
        COURT_STAGE,
        {"span": Precision(span_correct, span_total), "normalization": Precision(norm_correct, norm_total)},
    )


def score_dates(document: Document) -> StageScore:
    """Score every full occurrence's date outcome, including absence."""
    checkpoint = document.get_stage(DATE_STAGE)
    gold = _gold(_rows(checkpoint))
    span_correct = span_total = norm_correct = norm_total = 0
    for citation in checkpoint.full_locators:
        row = gold.get(_key(citation))
        reading = citation.date[-1] if citation.date else None
        span_total += 1
        span_correct += int(row is not None and _source_agrees(reading, _target(row, "date")))
        norm_total += 1
        norm_correct += int(_normalization_agrees("date", reading, row))
    return StageScore(
        DATE_STAGE,
        {"span": Precision(span_correct, span_total), "normalization": Precision(norm_correct, norm_total)},
    )


def score_pin_cites(document: Document) -> StageScore:
    """Score every full occurrence's pin outcome, including absence."""
    checkpoint = document.get_stage(PIN_STAGE)
    gold = _gold(_rows(checkpoint))
    span_correct = span_total = norm_correct = norm_total = 0
    for citation in checkpoint.full_locators:
        row = gold.get(_key(citation))
        reading = citation.pin_cite[-1] if citation.pin_cite is not None else None
        span_total += 1
        span_correct += int(row is not None and _source_agrees(reading, _target(row, "pin_cite")))
        norm_total += 1
        norm_correct += int(_normalization_agrees("pin_cite", reading, row))
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


def score_docket_root_llm_reassignment(document: Document) -> StageScore:
    """Score root assignments for docket roots whose identity was reviewed."""
    checkpoint = document.get_stage(ROOT_REVIEW_STAGE)
    gold = _gold(_rows(checkpoint))
    by_id = {citation.id: citation for citation in checkpoint.full_locators}
    available_gold_ids = {
        row["id"] for citation in checkpoint.full_locators if (row := gold.get(_key(citation))) is not None
    }
    correct = total = 0
    for citation in checkpoint.full_locators:
        if not isinstance(citation, FullDocketCitation):
            continue
        review_nodes = {node.id for node in citation.nodes if node.stage == ROOT_REVIEW_STAGE}
        for review in citation.docket_root_reviews:
            if review.node_id not in review_nodes or review.decision is None:
                continue
            for candidate_id in review.candidate_ids:
                candidate = by_id[candidate_id]
                total += 1
                row = gold.get(_key(candidate))
                root = by_id.get(latest(candidate.root_id))
                root_row = gold.get(_key(root)) if root is not None else None
                if row is None or root_row is None:
                    continue
                if row["root_id"] in available_gold_ids:
                    correct += int(root_row["id"] == row["root_id"])
                else:
                    correct += int(root_row["root_id"] == row["root_id"])
    return StageScore(ROOT_REVIEW_STAGE, {"root_assignment": Precision(correct, total)})


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
    (ROOT_REVIEW_STAGE, score_docket_root_llm_reassignment),
)


def _score_root_fields(document: Document) -> dict[str, dict[str, FieldScore | RecallScore]]:
    """Use the same root-field comparisons at any explicit Document checkpoint."""
    gold = _gold(tuple(row for row in _rows(document) if row.get("is_root")))
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
    result: dict[str, dict[str, FieldScore | RecallScore]] = {}
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
        # Reviewed absence and unavailability count once, just like a value.
        span_gold = norm_gold = len(gold_rows)
        span_correct = span_predicted = norm_correct = norm_predicted = 0
        for citation in document.roots:
            if gold_kind is not None and _kind(citation) != gold_kind:
                continue
            if name == "docket_entry" and not isinstance(citation, FullDocketCitation):
                continue
            readings = getattr(citation, attributes[name])
            reading = readings[-1] if readings else None
            row = gold.get(_key(citation))
            span_predicted += 1
            span_correct += int(row is not None and _source_agrees(reading, _target(row, target_name)))
            norm_predicted += 1
            norm_correct += int(_normalization_agrees(target_name, reading, row))
        result[name] = {
            "span": FieldScore(span_correct, span_predicted, span_gold),
            "span_overlap": _root_field_overlap_recall(
                document.roots, gold_rows, attribute=attributes[name], target_name=target_name
            ),
            "normalization": FieldScore(norm_correct, norm_predicted, norm_gold),
        }
        if name == "docket_locator":
            result["overall_locator"] = {
                measure: result["full_reporter_locator"][measure] + result["docket_locator"][measure]
                for measure in ("span", "span_overlap", "normalization")
            }
    return result


def score_grow_roots(document: Document) -> WorkflowScore:
    """Score extraction at its endpoint and, when present, validated root readings."""
    final_stage = ROOT_REVIEW_STAGE if ROOT_REVIEW_STAGE in document.stage_runs else ROOT_STAGE
    final = document.get_stage(final_stage)
    missing = [
        stage
        for stage, _ in GROW_ROOTS_STAGES
        if stage not in {HUNT_STAGE, ROOT_REVIEW_STAGE} and stage not in final.stage_runs
    ]
    if missing:
        raise ValueError(f"Incomplete grow_roots workflow; missing stages: {', '.join(missing)}")
    stages = tuple(score(final) for stage, score in GROW_ROOTS_STAGES if stage in final.stage_runs)
    validated = None
    if VALIDATED_FIELDS_STAGE in document.stage_runs:
        validated_fields = _score_root_fields(document.get_stage(VALIDATED_FIELDS_STAGE))
        validated = {name: validated_fields[name] for name in ("case_name", "court", "date")}
    return WorkflowScore(stages, _score_root_fields(final), validated)


def _precision_cell(value: Precision) -> str:
    return "—" if value.total == 0 else f"{value.correct}/{value.total} ({value.correct / value.total:.1%})"


def _field_cell(value: FieldScore, *, recall: bool) -> str:
    denominator = value.gold if recall else value.predicted
    return "—" if denominator == 0 else f"{value.correct}/{denominator} ({value.correct / denominator:.1%})"


def _recall_cell(value: RecallScore) -> str:
    return "—" if value.gold == 0 else f"{value.correct}/{value.gold} ({value.correct / value.gold:.1%})"


def _require_stage(score: StageScore, stage: str) -> None:
    if score.stage != stage:
        raise ValueError(f"Expected {stage} score, got {score.stage}")


def render_full_reporter_locators(score: StageScore) -> str:
    _require_stage(score, REPORTER_STAGE)
    return "\n".join(
        (
            f"## {REPORTER_STAGE}",
            "",
            "| Span precision | Normalization precision |",
            "| ---: | ---: |",
            f"| {_precision_cell(score.metrics['span'])} | {_precision_cell(score.metrics['normalization'])} |",
        )
    )


def render_docket_locators(score: StageScore) -> str:
    _require_stage(score, DOCKET_STAGE)
    return "\n".join(
        (
            f"## {DOCKET_STAGE}",
            "",
            "| Span precision | Normalization precision |",
            "| ---: | ---: |",
            f"| {_precision_cell(score.metrics['span'])} | {_precision_cell(score.metrics['normalization'])} |",
        )
    )


def render_docket_locator_site_hunting(score: StageScore) -> str:
    _require_stage(score, HUNT_STAGE)
    return "\n".join(
        (
            f"## {HUNT_STAGE}",
            "",
            "| Span precision | Normalization precision |",
            "| ---: | ---: |",
            f"| {_precision_cell(score.metrics['span'])} | {_precision_cell(score.metrics['normalization'])} |",
        )
    )


def render_docket_entries(score: StageScore) -> str:
    _require_stage(score, ENTRY_STAGE)
    return "\n".join(
        (
            f"## {ENTRY_STAGE}",
            "",
            "| Span precision | Normalization precision |",
            "| ---: | ---: |",
            f"| {_precision_cell(score.metrics['span'])} | {_precision_cell(score.metrics['normalization'])} |",
        )
    )


def render_colocations(score: StageScore) -> str:
    _require_stage(score, COLOCATION_STAGE)
    return "\n".join(
        (
            f"## {COLOCATION_STAGE}",
            "",
            "| Group precision |",
            "| ---: |",
            f"| {_precision_cell(score.metrics['groups'])} |",
        )
    )


def render_case_names(score: StageScore) -> str:
    _require_stage(score, CASE_NAME_STAGE)
    return "\n".join(
        (
            f"## {CASE_NAME_STAGE}",
            "",
            "| Span precision | Normalization precision |",
            "| ---: | ---: |",
            f"| {_precision_cell(score.metrics['span'])} | {_precision_cell(score.metrics['normalization'])} |",
        )
    )


def render_courts(score: StageScore) -> str:
    _require_stage(score, COURT_STAGE)
    return "\n".join(
        (
            f"## {COURT_STAGE}",
            "",
            "| Span precision | Normalization precision |",
            "| ---: | ---: |",
            f"| {_precision_cell(score.metrics['span'])} | {_precision_cell(score.metrics['normalization'])} |",
        )
    )


def render_dates(score: StageScore) -> str:
    _require_stage(score, DATE_STAGE)
    return "\n".join(
        (
            f"## {DATE_STAGE}",
            "",
            "| Span precision | Normalization precision |",
            "| ---: | ---: |",
            f"| {_precision_cell(score.metrics['span'])} | {_precision_cell(score.metrics['normalization'])} |",
        )
    )


def render_pin_cites(score: StageScore) -> str:
    _require_stage(score, PIN_STAGE)
    return "\n".join(
        (
            f"## {PIN_STAGE}",
            "",
            "| Span precision | Normalization precision |",
            "| ---: | ---: |",
            f"| {_precision_cell(score.metrics['span'])} | {_precision_cell(score.metrics['normalization'])} |",
        )
    )


def render_roots(score: StageScore) -> str:
    _require_stage(score, ROOT_STAGE)
    return "\n".join(
        (
            f"## {ROOT_STAGE}",
            "",
            "| Root assignment precision |",
            "| ---: |",
            f"| {_precision_cell(score.metrics['root_assignment'])} |",
        )
    )


def render_docket_root_llm_reassignment(score: StageScore) -> str:
    _require_stage(score, ROOT_REVIEW_STAGE)
    return "\n".join(
        (
            f"## {ROOT_REVIEW_STAGE}",
            "",
            "| Reviewed root assignment precision |",
            "| ---: |",
            f"| {_precision_cell(score.metrics['root_assignment'])} |",
        )
    )


GROW_ROOTS_RENDERERS: dict[str, Callable[[StageScore], str]] = {
    REPORTER_STAGE: render_full_reporter_locators,
    DOCKET_STAGE: render_docket_locators,
    HUNT_STAGE: render_docket_locator_site_hunting,
    ENTRY_STAGE: render_docket_entries,
    COLOCATION_STAGE: render_colocations,
    CASE_NAME_STAGE: render_case_names,
    COURT_STAGE: render_courts,
    DATE_STAGE: render_dates,
    PIN_STAGE: render_pin_cites,
    ROOT_STAGE: render_roots,
    ROOT_REVIEW_STAGE: render_docket_root_llm_reassignment,
}


def _render_root_fields(fields: dict[str, dict[str, FieldScore | RecallScore]], heading: str) -> str:
    lines = [
        heading,
        "",
        "| Field | Source/span precision | Source/span recall | Source/span overlap recall | Normalization precision | Normalization recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, measures in fields.items():
        span, overlap, norm = measures["span"], measures["span_overlap"], measures["normalization"]
        if (
            not isinstance(span, FieldScore)
            or not isinstance(overlap, RecallScore)
            or not isinstance(norm, FieldScore)
        ):
            raise ValueError("Root field score has an invalid measure type")
        label = "**overall_locator subtotal**" if name == "overall_locator" else name
        lines.append(
            f"| {label} | {_field_cell(span, recall=False)} | {_field_cell(span, recall=True)} | "
            f"{_recall_cell(overlap)} | {_field_cell(norm, recall=False)} | {_field_cell(norm, recall=True)} |"
        )
    return "\n".join(lines)


def render_grow_roots(
    score: WorkflowScore, *, include_stages: bool = True, set_name: str | None = None
) -> str:
    """Render the workflow; include every completed stage in order by default."""
    stage_names = tuple(item.stage for item in score.stages)
    expected = tuple(
        stage
        for stage, _ in GROW_ROOTS_STAGES
        if stage not in {HUNT_STAGE, ROOT_REVIEW_STAGE} or stage in stage_names
    )
    if stage_names != expected:
        raise ValueError("Grow-roots stage scores are missing or out of order")
    title = "# Grow-roots evaluation" + (f": {set_name}" if set_name else "")
    sections = [
        title,
        "Field-reading stage precision counts every eligible citation's field outcome, including absence, inference, and failed normalization; missing explicit gold raises.",
        "Docket site hunting: " + ("included" if HUNT_STAGE in stage_names else "not run"),
        "Docket root review: " + ("included" if ROOT_REVIEW_STAGE in stage_names else "not run"),
    ]
    if include_stages:
        sections.extend(GROW_ROOTS_RENDERERS[item.stage](item) for item in score.stages)
    sections.append(_render_root_fields(score.root_fields, "## Root fields"))
    if score.validated_root_fields is not None:
        if tuple(score.validated_root_fields) != ("case_name", "court", "date"):
            raise ValueError("Validated root fields must contain case name, court, and date")
        sections.append(
            _render_root_fields(
                score.validated_root_fields,
                f"## Root fields after {VALIDATED_FIELDS_STAGE}",
            )
        )
    return "\n\n".join(sections) + "\n"
