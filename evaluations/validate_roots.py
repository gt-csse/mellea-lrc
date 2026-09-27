"""Score the field judgments made by each root-validation stage."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mellea_lrc.model import Document, FullDocketCitation, FullReporterCitation
from mellea_lrc.model.citations.docket_lookup import DocketLookupReviewDecision
from mellea_lrc.model.citations.full import FullCitation
from mellea_lrc.model.citations.judgments import MatchResult
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactAmbiguityOutcome,
    ReporterExactLookupOutcome,
)
from mellea_lrc.validation.docket_root_lookup import STAGE as DOCKET_ROOT_LOOKUP
from mellea_lrc.validation.docket_root_lookup_review import STAGE as DOCKET_ROOT_LOOKUP_REVIEW
from mellea_lrc.validation.reporter_root_lookup import STAGE as REPORTER_ROOT_LOOKUP
from mellea_lrc.validation.reporter_root_lookup_ambiguous import STAGE as REPORTER_ROOT_LOOKUP_AMBIGUOUS
from mellea_lrc.validation.reporter_root_lookup_ambiguous_llm import (
    STAGE as REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM,
)
from mellea_lrc.validation.reporter_root_lookup_unique_llm import STAGE as REPORTER_ROOT_LOOKUP_UNIQUE_LLM

FIELDS = ("case_name", "court", "date")
GOLD_LABELS = frozenset({"agrees", "disagrees", "not_stated"})
STAGES = (
    REPORTER_ROOT_LOOKUP,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS,
    REPORTER_ROOT_LOOKUP_UNIQUE_LLM,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM,
    DOCKET_ROOT_LOOKUP_REVIEW,
)
WORKFLOW_STAGES = (*STAGES[:-1], DOCKET_ROOT_LOOKUP, STAGES[-1])


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
            self.correct + other.correct,
            self.predicted + other.predicted,
            self.gold + other.gold,
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
            raise ValueError("Cannot combine different validation stages")
        return StageScore(
            self.stage,
            {field: score + other.metrics[field] for field, score in self.metrics.items()},
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "metrics": {field: score.as_dict() for field, score in self.metrics.items()},
        }


@dataclass(frozen=True)
class WorkflowScore:
    stages: tuple[StageScore, ...]
    fields: dict[str, FieldScore]

    def __add__(self, other: WorkflowScore) -> WorkflowScore:
        if tuple(stage.stage for stage in self.stages) != tuple(stage.stage for stage in other.stages):
            raise ValueError("Cannot combine different validation workflows")
        if self.fields.keys() != other.fields.keys():
            raise ValueError("Cannot combine different validation fields")
        return WorkflowScore(
            tuple(left + right for left, right in zip(self.stages, other.stages, strict=True)),
            {field: score + other.fields[field] for field, score in self.fields.items()},
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "stages": [stage.as_dict() for stage in self.stages],
            "fields": {field: score.as_dict() for field, score in self.fields.items()},
        }


@dataclass(frozen=True)
class _GoldRoot:
    kind: str
    start: int
    end: int
    labels: dict[str, str]


def _gold_roots(document: Document) -> tuple[_GoldRoot, ...]:
    """Read every annotated root; missing field gold is an error, not a miss."""
    if document.source_path is None:
        raise ValueError("Validation evaluation needs an official source path")
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
        raise ValueError(f"Annotation does not match Document source: {annotation}")
    roots: list[_GoldRoot] = []
    spans: set[tuple[str, int, int]] = set()
    for line in lines[1:]:
        row = json.loads(line)
        if not row.get("is_root") or row.get("kind") not in {"FullCaseCitation", "DocketCitation"}:
            continue
        locator_field = row.get("locator")
        if not isinstance(locator_field, dict) or not isinstance(locator_field.get("source"), dict):
            raise ValueError(f"{row.get('id')}: root locator has no quoted source")
        locator = locator_field["source"]
        if "start" not in locator or "end" not in locator:
            raise ValueError(f"{row.get('id')}: root locator has no span")
        start, end = locator["start"], locator["end"]
        identity = (row.get("validation") or {}).get("identity") or {}
        labels = identity.get("fields") or {}
        if set(labels) != set(FIELDS) or any(
            not isinstance(labels[field], dict) or labels[field].get("label") not in GOLD_LABELS
            for field in FIELDS
        ):
            raise ValueError(f"{row.get('id')}: incomplete field-level identity gold")
        for field in FIELDS:
            if labels[field]["label"] == "not_stated" and ((row.get(field) or {}).get("source") or {}).get(
                "kind"
            ) not in {None, "not_stated"}:
                raise ValueError(f"{row.get('id')}: {field} gold is not_stated despite a source reading")
        key = (row["kind"], start, end)
        if key in spans:
            raise ValueError(f"Duplicate annotated root locator: {key}")
        spans.add(key)
        roots.append(
            _GoldRoot(
                row["kind"],
                start,
                end,
                {field: labels[field]["label"] for field in FIELDS},
            )
        )
    return tuple(roots)


def _kind(citation: FullCitation) -> str:
    if isinstance(citation, FullReporterCitation):
        return "FullCaseCitation"
    if isinstance(citation, FullDocketCitation):
        return "DocketCitation"
    raise ValueError(f"Unsupported validation root kind: {type(citation).__name__}")


def _align(roots: tuple[FullCitation, ...], gold: tuple[_GoldRoot, ...]) -> dict[int, int]:
    """Match roots once by exact locator, then by positive locator overlap."""
    by_exact = {(item.kind, item.start, item.end): index for index, item in enumerate(gold)}
    assigned: dict[int, int] = {}
    locked: set[int] = set()
    for prediction_index, root in enumerate(roots):
        span = root.locator_span
        gold_index = by_exact.get((_kind(root), span.start, span.end))
        if gold_index is not None:
            assigned[gold_index] = prediction_index
            locked.add(gold_index)

    options: dict[int, list[int]] = {}
    for prediction_index, root in enumerate(roots):
        if prediction_index in assigned.values():
            continue
        span = root.locator_span
        overlapping = [
            gold_index
            for gold_index, item in enumerate(gold)
            if gold_index not in locked
            and item.kind == _kind(root)
            and span.start < item.end
            and item.start < span.end
        ]
        options[prediction_index] = sorted(
            overlapping,
            key=lambda index: -(min(span.end, gold[index].end) - max(span.start, gold[index].start)),
        )

    def claim(prediction_index: int, seen: set[int]) -> bool:
        for gold_index in options[prediction_index]:
            if gold_index in seen:
                continue
            seen.add(gold_index)
            previous = assigned.get(gold_index)
            if previous is None or claim(previous, seen):
                assigned[gold_index] = prediction_index
                return True
        return False

    for prediction_index in options:
        claim(prediction_index, set())
    return {prediction_index: gold_index for gold_index, prediction_index in assigned.items()}


def _selected_candidate(root: FullReporterCitation, stage: str) -> int | None:
    if stage == REPORTER_ROOT_LOOKUP:
        lookup = root.reporter_exact_lookup
        return 0 if lookup is not None and lookup.outcome is ReporterExactLookupOutcome.UNIQUE else None
    if stage == REPORTER_ROOT_LOOKUP_AMBIGUOUS:
        resolution = root.reporter_exact_ambiguity_resolution
        return (
            resolution.selected_candidate_index
            if resolution is not None
            and resolution.outcome is ReporterExactAmbiguityOutcome.UNIQUE_RULE_MATCH
            else None
        )
    if stage == REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM:
        review = root.reporter_ambiguous_review
        return review.decision.selected_candidate_index if review is not None and review.decision else None
    if stage == REPORTER_ROOT_LOOKUP_UNIQUE_LLM:
        review = root.reporter_unique_review
        return 0 if review is not None and review.decision is not None else None
    raise ValueError(f"Unknown validation stage: {stage}")


def _label(result: MatchResult, *, source_present: bool) -> str:
    if result is MatchResult.MATCH:
        return "agrees"
    if result is MatchResult.MISMATCH:
        return "disagrees"
    if result is MatchResult.UNAVAILABLE:
        return "unavailable" if source_present else "not_stated"
    raise ValueError(f"Unknown field judgment: {result}")


def _selected_docket_review(root: FullDocketCitation) -> DocketLookupReviewDecision | None:
    review = root.docket_lookup_review
    if review is None or review.decision is None or review.decision.selected_candidate_index is None:
        return None
    if not any(node.id == review.node_id and node.stage == DOCKET_ROOT_LOOKUP_REVIEW for node in root.nodes):
        return None
    return review.decision


def score_docket_root_lookup_review(document: Document) -> StageScore:
    checkpoint = document.get_stage(DOCKET_ROOT_LOOKUP_REVIEW)
    gold = _gold_roots(checkpoint)
    aligned = _align(checkpoint.roots, gold)
    counts = {field: [0, 0] for field in FIELDS}
    for prediction_index, root in enumerate(checkpoint.roots):
        if not isinstance(root, FullDocketCitation) or (decision := _selected_docket_review(root)) is None:
            continue
        gold_root = gold[aligned[prediction_index]] if prediction_index in aligned else None
        for field in FIELDS:
            result = getattr(decision, field).result
            label = _label(result, source_present=bool(getattr(root, field)))
            counts[field][1] += 1
            counts[field][0] += int(gold_root is not None and label == gold_root.labels[field])
    return StageScore(
        DOCKET_ROOT_LOOKUP_REVIEW,
        {field: Precision(*counts[field]) for field in FIELDS},
    )


def _stage_judgments(
    document: Document,
    stage: str,
    candidate: Callable[[FullReporterCitation], int | None],
) -> StageScore:
    checkpoint = document.get_stage(stage)
    gold = _gold_roots(checkpoint)
    aligned = _align(checkpoint.roots, gold)
    counts = {field: [0, 0] for field in FIELDS}
    for prediction_index, root in enumerate(checkpoint.roots):
        if not isinstance(root, FullReporterCitation):
            continue
        node_ids = {node.id for node in root.nodes if node.stage == stage}
        if not node_ids or (candidate_index := candidate(root)) is None:
            continue
        gold_root = gold[aligned[prediction_index]] if prediction_index in aligned else None
        for field in FIELDS:
            decisions = [
                judgment
                for judgment in getattr(root, f"{field}_judgments")
                if judgment.node_id in node_ids and judgment.candidate_index == candidate_index
            ]
            if len(decisions) > 1:
                raise ValueError("A stage produced duplicate selected-candidate field judgments")
            if not decisions:
                continue
            counts[field][1] += 1
            label = _label(decisions[0].result, source_present=bool(getattr(root, field)))
            counts[field][0] += int(gold_root is not None and label == gold_root.labels[field])
    return StageScore(stage, {field: Precision(*counts[field]) for field in FIELDS})


def score_reporter_root_lookup(document: Document) -> StageScore:
    return _stage_judgments(
        document,
        REPORTER_ROOT_LOOKUP,
        lambda root: _selected_candidate(root, REPORTER_ROOT_LOOKUP),
    )


def score_reporter_root_lookup_ambiguous(document: Document) -> StageScore:
    return _stage_judgments(
        document,
        REPORTER_ROOT_LOOKUP_AMBIGUOUS,
        lambda root: _selected_candidate(root, REPORTER_ROOT_LOOKUP_AMBIGUOUS),
    )


def score_reporter_root_lookup_ambiguous_llm(document: Document) -> StageScore:
    return _stage_judgments(
        document,
        REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM,
        lambda root: _selected_candidate(root, REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM),
    )


def score_reporter_root_lookup_unique_llm(document: Document) -> StageScore:
    return _stage_judgments(
        document,
        REPORTER_ROOT_LOOKUP_UNIQUE_LLM,
        lambda root: _selected_candidate(root, REPORTER_ROOT_LOOKUP_UNIQUE_LLM),
    )


STAGE_SCORERS: tuple[tuple[str, Callable[[Document], StageScore]], ...] = (
    (REPORTER_ROOT_LOOKUP, score_reporter_root_lookup),
    (REPORTER_ROOT_LOOKUP_AMBIGUOUS, score_reporter_root_lookup_ambiguous),
    (REPORTER_ROOT_LOOKUP_UNIQUE_LLM, score_reporter_root_lookup_unique_llm),
    (REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM, score_reporter_root_lookup_ambiguous_llm),
    (DOCKET_ROOT_LOOKUP_REVIEW, score_docket_root_lookup_review),
)


def _final_reporter_field_label(
    root: FullReporterCitation, field: str, stage_runs: tuple[str, ...]
) -> str | None:
    selected_stages = [
        (stage, candidate_index)
        for stage in stage_runs
        if stage in STAGES and stage != DOCKET_ROOT_LOOKUP_REVIEW
        if (candidate_index := _selected_candidate(root, stage)) is not None
    ]
    if not selected_stages:
        return None
    if not getattr(root, field):
        return "not_stated"
    for stage, candidate_index in reversed(selected_stages):
        node_ids = {node.id for node in root.nodes if node.stage == stage}
        selected = [
            judgment
            for judgment in getattr(root, f"{field}_judgments")
            if judgment.node_id in node_ids and judgment.candidate_index == candidate_index
        ]
        if len(selected) > 1:
            raise ValueError("A stage produced duplicate selected-candidate field judgments")
        if selected:
            return _label(selected[0].result, source_present=True)
    return None


def _final_docket_field_label(root: FullDocketCitation, field: str) -> str | None:
    decision = _selected_docket_review(root)
    if decision is None:
        return None
    return _label(getattr(decision, field).result, source_present=bool(getattr(root, field)))


def _final_field_label(root: FullCitation, field: str, stage_runs: tuple[str, ...]) -> str | None:
    if isinstance(root, FullDocketCitation):
        return _final_docket_field_label(root, field)
    if isinstance(root, FullReporterCitation):
        return _final_reporter_field_label(root, field, stage_runs)
    raise ValueError(f"Unsupported validation root kind: {type(root).__name__}")


def score_validate_roots(document: Document) -> WorkflowScore:
    """Score completed validation judgments and every annotated root's final fields."""
    if any(stage not in document.stage_runs for stage in WORKFLOW_STAGES):
        missing = [stage for stage in WORKFLOW_STAGES if stage not in document.stage_runs]
        raise ValueError(f"Incomplete validate_roots workflow; missing stages: {', '.join(missing)}")
    final_stage = max(WORKFLOW_STAGES, key=document.stage_runs.index)
    final = document.get_stage(final_stage)
    stage_scores = tuple(score(final) for _, score in STAGE_SCORERS)
    gold = _gold_roots(final)
    aligned = _align(final.roots, gold)
    counts = {field: [0, 0] for field in FIELDS}
    for prediction_index, root in enumerate(final.roots):
        if isinstance(root, FullReporterCitation) and not any(node.stage in STAGES for node in root.nodes):
            continue
        gold_root = gold[aligned[prediction_index]] if prediction_index in aligned else None
        for field in FIELDS:
            outcome = _final_field_label(root, field, final.stage_runs)
            if outcome is None:
                continue
            counts[field][1] += 1
            counts[field][0] += int(gold_root is not None and outcome == gold_root.labels[field])
    return WorkflowScore(
        stage_scores,
        {field: FieldScore(counts[field][0], counts[field][1], len(gold)) for field in FIELDS},
    )


def _precision_cell(score: Precision) -> str:
    return "—" if score.total == 0 else f"{score.correct}/{score.total} ({score.correct / score.total:.1%})"


def _field_cell(score: FieldScore, *, recall: bool) -> str:
    denominator = score.gold if recall else score.predicted
    return "—" if denominator == 0 else f"{score.correct}/{denominator} ({score.correct / denominator:.1%})"


def _render_stage(score: StageScore, expected: str) -> str:
    if score.stage != expected or tuple(score.metrics) != FIELDS:
        raise ValueError(f"Expected {expected} field score")
    lines = [
        f"## {expected}",
        "",
        "| Field | Precision |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {field} | {_precision_cell(score.metrics[field])} |" for field in FIELDS)
    return "\n".join(lines)


def render_reporter_root_lookup(score: StageScore) -> str:
    return _render_stage(score, REPORTER_ROOT_LOOKUP) + "\n"


def render_docket_root_lookup_review(score: StageScore) -> str:
    return _render_stage(score, DOCKET_ROOT_LOOKUP_REVIEW) + "\n"


def render_reporter_root_lookup_ambiguous(score: StageScore) -> str:
    return _render_stage(score, REPORTER_ROOT_LOOKUP_AMBIGUOUS) + "\n"


def render_reporter_root_lookup_ambiguous_llm(score: StageScore) -> str:
    return _render_stage(score, REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM) + "\n"


def render_reporter_root_lookup_unique_llm(score: StageScore) -> str:
    return _render_stage(score, REPORTER_ROOT_LOOKUP_UNIQUE_LLM) + "\n"


def render_validate_roots(
    score: WorkflowScore,
    *,
    set_name: str | None = None,
    include_stages: bool = True,
) -> str:
    if tuple(stage.stage for stage in score.stages) != STAGES or tuple(score.fields) != FIELDS:
        raise ValueError("Validate-roots score has missing or out-of-order stages or fields")
    sections = [f"# Validate-roots evaluation{f': {set_name}' if set_name else ''}"]
    if include_stages:
        sections.extend(_render_stage(stage, stage.stage) for stage in score.stages)
    lines = [
        "## Root field judgments",
        "",
        "| Field | Precision | Recall |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {field} | {_field_cell(score.fields[field], recall=False)} | "
        f"{_field_cell(score.fields[field], recall=True)} |"
        for field in FIELDS
    )
    sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"
