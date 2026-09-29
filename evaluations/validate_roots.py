"""Score field and identity judgments through the root-lookup workflow."""

from __future__ import annotations

import hashlib
import html
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from mellea_lrc.model import Document, FullDocketCitation, FullReporterCitation
from mellea_lrc.model.citations.docket_lookup import DocketLookupReviewDecision
from mellea_lrc.model.citations.full import FullCitation
from mellea_lrc.model.citations.judgments import IdentityVerdict, MatchResult
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactAmbiguityOutcome,
    ReporterExactLookupOutcome,
)
from mellea_lrc.model.span import Span
from mellea_lrc.validation.body_search.courtlistener_opinion import STAGE as COURTLISTENER_OPINION_BODY_SEARCH
from mellea_lrc.validation.body_search.courtlistener_recap import STAGE as COURTLISTENER_RECAP_BODY_SEARCH
from mellea_lrc.validation.body_search.govinfo import STAGE as GOVINFO_OPINION_BODY_SEARCH
from mellea_lrc.validation.docket_root_lookup import STAGE as DOCKET_ROOT_LOOKUP
from mellea_lrc.validation.docket_root_lookup_review import STAGE as DOCKET_ROOT_LOOKUP_REVIEW
from mellea_lrc.validation.govinfo_docket_lookup import STAGE as GOVINFO_DOCKET_LOOKUP
from mellea_lrc.validation.govinfo_docket_lookup_review import STAGE as GOVINFO_DOCKET_LOOKUP_REVIEW
from mellea_lrc.validation.locator_body_review import STAGE as LOCATOR_BODY_REVIEW
from mellea_lrc.validation.reporter_root_lookup import STAGE as REPORTER_ROOT_LOOKUP
from mellea_lrc.validation.reporter_root_lookup_ambiguous import STAGE as REPORTER_ROOT_LOOKUP_AMBIGUOUS
from mellea_lrc.validation.reporter_root_lookup_ambiguous_llm import (
    STAGE as REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM,
)
from mellea_lrc.validation.reporter_root_lookup_unique_llm import STAGE as REPORTER_ROOT_LOOKUP_UNIQUE_LLM

FIELDS = ("case_name", "court", "date")
GOLD_LABELS = frozenset({"agrees", "disagrees", "not_stated"})
GOLD_IDENTITIES = frozenset({"CORRECT_IDENTITY", "WRONG_IDENTITY"})
STAGES = (
    REPORTER_ROOT_LOOKUP,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS,
    REPORTER_ROOT_LOOKUP_UNIQUE_LLM,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM,
    DOCKET_ROOT_LOOKUP_REVIEW,
    GOVINFO_DOCKET_LOOKUP_REVIEW,
)
WORKFLOW_STAGES = (
    *STAGES[:-2],
    DOCKET_ROOT_LOOKUP,
    DOCKET_ROOT_LOOKUP_REVIEW,
    GOVINFO_DOCKET_LOOKUP,
    STAGES[-1],
)
BODY_WORKFLOW_STAGES = (
    COURTLISTENER_OPINION_BODY_SEARCH,
    COURTLISTENER_RECAP_BODY_SEARCH,
    GOVINFO_OPINION_BODY_SEARCH,
    LOCATOR_BODY_REVIEW,
)


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
class IdentityScore:
    correct: int = 0
    predicted: int = 0
    gold: int = 0
    undetermined: int = 0

    def __add__(self, other: IdentityScore) -> IdentityScore:
        return IdentityScore(
            self.correct + other.correct,
            self.predicted + other.predicted,
            self.gold + other.gold,
            self.undetermined + other.undetermined,
        )

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "correct": self.correct,
            "predicted": self.predicted,
            "gold": self.gold,
            "undetermined": self.undetermined,
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
class BodyReviewRow:
    """One stage-23 review with the evidence behind its judgment or route."""

    source_path: str
    citation_id: str
    citation_kind: str
    locator_start: int
    locator_end: int
    locator_quote: str
    gold_identity: str | None
    review_status: str
    next_stage: str | None
    selected_source: str | None
    selected_evidence_index: int | None
    selected_body_id: str | None
    selected_body_url: str | None
    selected_source_offset: int | None
    grounded_citation_quote: str | None
    grounded_citation_span: dict[str, int] | None
    grounded_context_quote: str | None
    grounded_context_span: dict[str, int] | None
    treatment: str | None
    filing: dict[str, str | None] | None
    third_party: dict[str, str | None] | None
    comparisons: dict[str, dict[str, str]] | None
    verdict: str | None
    reason: str | None
    failure_reason: str | None
    matches_gold: bool | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BodyReviewScore:
    """Stage-23 outcomes and review details, scored only where binary gold exists."""

    stage: str
    decisive_identity: Precision
    verdict_counts: dict[str, int]
    route_counts: dict[str, int]
    review_status_counts: dict[str, int]
    rows: tuple[BodyReviewRow, ...]

    def __add__(self, other: BodyReviewScore) -> BodyReviewScore:
        if self.stage != other.stage:
            raise ValueError("Cannot combine different body-review stages")

        def combine(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
            return {key: left.get(key, 0) + right.get(key, 0) for key in left.keys() | right.keys()}

        return BodyReviewScore(
            self.stage,
            self.decisive_identity + other.decisive_identity,
            combine(self.verdict_counts, other.verdict_counts),
            combine(self.route_counts, other.route_counts),
            combine(self.review_status_counts, other.review_status_counts),
            (*self.rows, *other.rows),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "decisive_identity": self.decisive_identity.as_dict(),
            "verdict_counts": dict(sorted(self.verdict_counts.items())),
            "route_counts": dict(sorted(self.route_counts.items())),
            "review_status_counts": dict(sorted(self.review_status_counts.items())),
            "rows": [row.as_dict() for row in self.rows],
        }


@dataclass(frozen=True)
class WorkflowScore:
    stages: tuple[StageScore, ...]
    fields: dict[str, FieldScore]
    identity: IdentityScore
    checkpoint: str | None = None
    body_review: BodyReviewScore | None = None

    def __add__(self, other: WorkflowScore) -> WorkflowScore:
        if tuple(stage.stage for stage in self.stages) != tuple(stage.stage for stage in other.stages):
            raise ValueError("Cannot combine different validation workflows")
        if self.fields.keys() != other.fields.keys():
            raise ValueError("Cannot combine different validation fields")
        if self.checkpoint != other.checkpoint:
            raise ValueError("Cannot combine different validation checkpoints")
        if (self.body_review is None) != (other.body_review is None):
            raise ValueError("Cannot combine workflows with different body-review details")
        return WorkflowScore(
            tuple(left + right for left, right in zip(self.stages, other.stages, strict=True)),
            {field: score + other.fields[field] for field, score in self.fields.items()},
            self.identity + other.identity,
            self.checkpoint,
            (
                self.body_review + other.body_review
                if self.body_review is not None and other.body_review is not None
                else None
            ),
        )

    def as_dict(self) -> dict[str, object]:
        result = {
            "stages": [stage.as_dict() for stage in self.stages],
            "fields": {field: score.as_dict() for field, score in self.fields.items()},
            "identity": self.identity.as_dict(),
        }
        if self.checkpoint is not None:
            result["checkpoint"] = self.checkpoint
        if self.body_review is not None:
            result["body_review"] = self.body_review.as_dict()
        return result


@dataclass(frozen=True)
class _GoldRoot:
    kind: str
    start: int
    end: int
    labels: dict[str, str]
    identity: str


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
        if identity.get("label") not in GOLD_IDENTITIES:
            raise ValueError(f"{row.get('id')}: missing or invalid root identity gold")
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
                identity["label"],
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


def _selected_govinfo_docket_review(root: FullDocketCitation) -> DocketLookupReviewDecision | None:
    review = root.govinfo_docket_review
    lookup = root.govinfo_docket_lookup
    if (
        review is None
        or review.decision is None
        or review.decision.selected_candidate_index is None
        or lookup is None
        or review.decision.selected_candidate_index not in lookup.shortlisted_candidate_indices
    ):
        return None
    if not any(
        node.id == lookup.node_id and node.stage == GOVINFO_DOCKET_LOOKUP for node in root.nodes
    ) or not any(
        node.id == review.node_id and node.stage == GOVINFO_DOCKET_LOOKUP_REVIEW for node in root.nodes
    ):
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


def score_govinfo_docket_lookup_review(document: Document) -> StageScore:
    checkpoint = document.get_stage(GOVINFO_DOCKET_LOOKUP_REVIEW)
    gold = _gold_roots(checkpoint)
    aligned = _align(checkpoint.roots, gold)
    counts = {field: [0, 0] for field in FIELDS}
    for prediction_index, root in enumerate(checkpoint.roots):
        if (
            not isinstance(root, FullDocketCitation)
            or (decision := _selected_govinfo_docket_review(root)) is None
        ):
            continue
        gold_root = gold[aligned[prediction_index]] if prediction_index in aligned else None
        for field in FIELDS:
            result = getattr(decision, field).result
            counts[field][1] += 1
            label = _label(result, source_present=bool(getattr(root, field)))
            counts[field][0] += int(gold_root is not None and label == gold_root.labels[field])
    return StageScore(
        GOVINFO_DOCKET_LOOKUP_REVIEW,
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
    (GOVINFO_DOCKET_LOOKUP_REVIEW, score_govinfo_docket_lookup_review),
)


def _final_reporter_field_label(
    root: FullReporterCitation, field: str, stage_runs: tuple[str, ...]
) -> str | None:
    selected_stages = [
        (stage, candidate_index)
        for stage in stage_runs
        if stage in STAGES[:-2]
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
    decision = _selected_govinfo_docket_review(root) or _selected_docket_review(root)
    if decision is None:
        return None
    return _label(getattr(decision, field).result, source_present=bool(getattr(root, field)))


def _final_field_label(root: FullCitation, field: str, stage_runs: tuple[str, ...]) -> str | None:
    if isinstance(root, FullDocketCitation):
        return _final_docket_field_label(root, field)
    if isinstance(root, FullReporterCitation):
        return _final_reporter_field_label(root, field, stage_runs)
    raise ValueError(f"Unsupported validation root kind: {type(root).__name__}")


def _identity_value(labels: dict[str, str]) -> str:
    """Resolve one selected lookup's field judgments without inventing missing evidence."""
    if set(labels) != set(FIELDS) or any(
        label not in {"agrees", "disagrees", "unavailable", "not_stated"} for label in labels.values()
    ):
        raise ValueError("Identity requires three explicit field judgments")
    if "disagrees" in labels.values():
        return "WRONG_IDENTITY"
    if labels["case_name"] in {"unavailable", "not_stated"}:
        return "UNDETERMINED"
    return "CORRECT_IDENTITY"


def _body_verdict(root: FullCitation) -> IdentityVerdict | None:
    if not root.body_reviews:
        return None
    review = root.body_reviews[-1]
    if not any(node.id == review.node_id and node.stage == LOCATOR_BODY_REVIEW for node in root.nodes):
        raise ValueError("Body review does not reference its stage node")
    judgments = [judgment for judgment in root.identity_judgments if judgment.node_id == review.node_id]
    selected = review.decision is not None and review.decision.source is not None
    if selected:
        if len(judgments) != 1 or judgments[0].verdict is not review.decision.identity_verdict:
            raise ValueError("Selected body review must issue its matching identity judgment")
        return judgments[0].verdict
    if judgments:
        raise ValueError("Body review without selected evidence cannot issue an identity judgment")
    return None


def _body_route(root: FullCitation, node_id: str) -> str | None:
    routes = [route for route in root.routes if route.node_id == node_id]
    if len(routes) != 1:
        raise ValueError("Body review must record exactly one route decision")
    return routes[0].value


def _span_dict(span: Span | None) -> dict[str, int] | None:
    return {"start": span.start, "end": span.end} if span is not None else None


def _body_field_values(fields: object) -> dict[str, str | None]:
    return {field: getattr(fields, field) for field in ("locator", *FIELDS)}


def score_locator_body_review(document: Document) -> BodyReviewScore:
    """Expose each stage-23 result, and score only decisive identity verdicts."""
    checkpoint = document.get_stage(LOCATOR_BODY_REVIEW)
    gold = _gold_roots(checkpoint)
    aligned = _align(checkpoint.roots, gold)
    correct = predicted = 0
    verdict_counts: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    review_status_counts: dict[str, int] = {}
    rows: list[BodyReviewRow] = []
    for index, root in enumerate(checkpoint.roots):
        stage_nodes = [node for node in root.nodes if node.stage == LOCATOR_BODY_REVIEW]
        if not stage_nodes:
            if root.body_reviews:
                raise ValueError("Body review does not reference its stage node")
            continue
        reviews = [
            review for review in root.body_reviews if review.node_id in {node.id for node in stage_nodes}
        ]
        if len(stage_nodes) != 1 or len(reviews) != 1:
            raise ValueError("Stage 23 must record exactly one body review for each processed root")
        review = reviews[0]
        decision = review.decision
        verdict = _body_verdict(root)
        route = _body_route(root, review.node_id)
        if (verdict is None or verdict.name not in GOLD_IDENTITIES) != (route is not None):
            raise ValueError("Unresolved body reviews must route; decisive verdicts must clear the route")
        if verdict is not None:
            verdict_counts[verdict.value] = verdict_counts.get(verdict.value, 0) + 1
        if route is not None:
            route_counts[route] = route_counts.get(route, 0) + 1
        review_status = (
            "review_failure"
            if decision is None
            else "no_reviewable_evidence"
            if decision.source is None
            and decision.reason == "No fetched third-party body citation is available for comparison."
            and review.ivr is None
            else "model_declined"
            if decision.source is None
            else "selected"
        )
        review_status_counts[review_status] = review_status_counts.get(review_status, 0) + 1
        gold_identity = gold[aligned[index]].identity if index in aligned else None
        matches_gold = None
        if verdict is not None and verdict.name in GOLD_IDENTITIES:
            predicted += 1
            matches_gold = verdict.name == gold_identity
            correct += int(matches_gold)

        selected_source = decision.source if decision is not None else None
        selected_body_id = selected_body_url = selected_source_offset = None
        if selected_source is not None:
            search = next((item for item in root.body_searches if item.source is selected_source), None)
            if (
                search is None
                or decision.evidence_index is None
                or decision.evidence_index >= len(search.evidence)
            ):
                raise ValueError("Body review selected evidence missing from its saved search")
            evidence = search.evidence[decision.evidence_index]
            selected_body_id, selected_body_url, selected_source_offset = (
                evidence.body_id,
                evidence.url,
                evidence.source_offset,
            )
        comparisons = (
            {
                field: {
                    "result": getattr(decision.comparisons, field).result.value,
                    "reason": getattr(decision.comparisons, field).reason,
                }
                for field in ("locator", *FIELDS)
            }
            if decision is not None and decision.comparisons is not None
            else None
        )
        span = root.locator_span
        rows.append(
            BodyReviewRow(
                source_path=str(checkpoint.source_path),
                citation_id=root.id,
                citation_kind=_kind(root),
                locator_start=span.start,
                locator_end=span.end,
                locator_quote=checkpoint.text[span.start : span.end],
                gold_identity=gold_identity,
                review_status=review_status,
                next_stage=route,
                selected_source=selected_source.value if selected_source is not None else None,
                selected_evidence_index=decision.evidence_index if decision is not None else None,
                selected_body_id=selected_body_id,
                selected_body_url=selected_body_url,
                selected_source_offset=selected_source_offset,
                grounded_citation_quote=review.grounded_quote,
                grounded_citation_span=_span_dict(review.quote_span),
                grounded_context_quote=review.grounded_context,
                grounded_context_span=_span_dict(review.context_span),
                treatment=(decision.treatment.value if decision is not None and decision.treatment else None),
                filing=(
                    _body_field_values(decision.filing)
                    if decision is not None and decision.filing is not None
                    else None
                ),
                third_party=(
                    _body_field_values(decision.third_party)
                    if decision is not None and decision.third_party is not None
                    else None
                ),
                comparisons=comparisons,
                verdict=verdict.value if verdict is not None else None,
                reason=decision.reason if decision is not None else None,
                failure_reason=review.failure_reason,
                matches_gold=matches_gold,
            )
        )
    return BodyReviewScore(
        LOCATOR_BODY_REVIEW,
        Precision(correct, predicted),
        verdict_counts,
        route_counts,
        review_status_counts,
        tuple(rows),
    )


def score_validate_roots(document: Document) -> WorkflowScore:
    """Score completed validation judgments and every annotated root's final fields."""
    if any(stage not in document.stage_runs for stage in WORKFLOW_STAGES):
        missing = [stage for stage in WORKFLOW_STAGES if stage not in document.stage_runs]
        raise ValueError(f"Incomplete validate_roots workflow; missing stages: {', '.join(missing)}")
    body_stages = set(BODY_WORKFLOW_STAGES).intersection(document.stage_runs)
    if body_stages and body_stages != set(BODY_WORKFLOW_STAGES):
        missing = [stage for stage in BODY_WORKFLOW_STAGES if stage not in document.stage_runs]
        raise ValueError(f"Incomplete locator-body workflow; missing stages: {', '.join(missing)}")
    judgment_stage = max(WORKFLOW_STAGES, key=document.stage_runs.index)
    final_stage = LOCATOR_BODY_REVIEW if body_stages else judgment_stage
    final = document.get_stage(final_stage)
    stage_scores = tuple(score(final) for _, score in STAGE_SCORERS)
    gold = _gold_roots(final)
    # Body review can correct extracted readings, but it does not judge whether
    # those fields identify the same case as an external validation candidate.
    judged = final.get_stage(judgment_stage)
    aligned = _align(judged.roots, gold)
    counts = {field: [0, 0] for field in FIELDS}
    identity_correct = identity_predicted = undetermined = 0
    lookup_verdicts: dict[str, str] = {}
    for prediction_index, root in enumerate(judged.roots):
        if isinstance(root, FullReporterCitation) and not any(node.stage in STAGES for node in root.nodes):
            continue
        gold_root = gold[aligned[prediction_index]] if prediction_index in aligned else None
        outcomes: dict[str, str] = {}
        for field in FIELDS:
            outcome = _final_field_label(root, field, judged.stage_runs)
            if outcome is None:
                continue
            outcomes[field] = outcome
            counts[field][1] += 1
            counts[field][0] += int(gold_root is not None and outcome == gold_root.labels[field])
        if not outcomes:
            continue
        verdict = _identity_value(outcomes)
        lookup_verdicts[root.id] = verdict
        if verdict == "UNDETERMINED":
            undetermined += 1
        else:
            identity_predicted += 1
            identity_correct += int(gold_root is not None and verdict == gold_root.identity)
    identity = IdentityScore(identity_correct, identity_predicted, len(gold), undetermined)
    body_review = None
    if body_stages:
        body_review = score_locator_body_review(final)
        final_aligned = _align(final.roots, gold)
        identity_correct = identity_predicted = undetermined = 0
        for prediction_index, root in enumerate(final.roots):
            body_verdict = _body_verdict(root)
            if body_verdict is not None and body_verdict.name in {
                "PARTIALLY_CORROBORATED",
                "UNDETERMINED",
            }:
                # A nondecisive body judgment retracts an earlier full-citation admission.
                verdict = "UNDETERMINED"
            elif body_verdict is not None and body_verdict.name in GOLD_IDENTITIES:
                verdict = body_verdict.name
            else:
                verdict = lookup_verdicts.get(root.id)
            if verdict is None:
                continue
            if verdict == "UNDETERMINED":
                undetermined += 1
                continue
            identity_predicted += 1
            identity_correct += int(
                prediction_index in final_aligned
                and verdict == gold[final_aligned[prediction_index]].identity
            )
        identity = IdentityScore(identity_correct, identity_predicted, len(gold), undetermined)
    return WorkflowScore(
        stage_scores,
        {field: FieldScore(counts[field][0], counts[field][1], len(gold)) for field in FIELDS},
        identity,
        LOCATOR_BODY_REVIEW if body_stages else None,
        body_review,
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


def render_govinfo_docket_lookup_review(score: StageScore) -> str:
    return _render_stage(score, GOVINFO_DOCKET_LOOKUP_REVIEW) + "\n"


def render_reporter_root_lookup_ambiguous(score: StageScore) -> str:
    return _render_stage(score, REPORTER_ROOT_LOOKUP_AMBIGUOUS) + "\n"


def render_reporter_root_lookup_ambiguous_llm(score: StageScore) -> str:
    return _render_stage(score, REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM) + "\n"


def render_reporter_root_lookup_unique_llm(score: StageScore) -> str:
    return _render_stage(score, REPORTER_ROOT_LOOKUP_UNIQUE_LLM) + "\n"


def _review_cell(value: object) -> str:
    if value is None:
        return "—"
    return html.escape(str(value)).replace("|", "&#124;").replace("\n", "<br>")


def _review_span_cell(span: dict[str, int] | None) -> str:
    return f"{span['start']}:{span['end']}" if span is not None else "—"


def render_locator_body_review(score: BodyReviewScore) -> str:
    """Render counts and the saved evidence for each stage-23 review."""
    if score.stage != LOCATOR_BODY_REVIEW:
        raise ValueError(f"Expected {LOCATOR_BODY_REVIEW} review score")
    lines = [
        f"## {LOCATOR_BODY_REVIEW}",
        "",
        "Only correct_identity and wrong_identity are scored against the annotated binary identity label. "
        "Printed-field comparisons have no corresponding gold and are shown for review only.",
        "",
        "| Decisive identity precision |",
        "| ---: |",
        f"| {_precision_cell(score.decisive_identity)} |",
        "",
        "### Issued verdicts",
        "",
        "| Verdict | Count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {_review_cell(verdict)} | {count} |" for verdict, count in sorted(score.verdict_counts.items())
    )
    if not score.verdict_counts:
        lines.append("| — | 0 |")
    lines.extend(("", "### Routes", "", "| Next stage | Count |", "| --- | ---: |"))
    lines.extend(
        f"| {_review_cell(route)} | {count} |" for route, count in sorted(score.route_counts.items())
    )
    if not score.route_counts:
        lines.append("| — | 0 |")
    lines.extend(("", "### Review status", "", "| Status | Count |", "| --- | ---: |"))
    lines.extend(
        f"| {_review_cell(status)} | {count} |"
        for status, count in sorted(score.review_status_counts.items())
    )
    if not score.review_status_counts:
        lines.append("| — | 0 |")
    lines.extend(
        (
            "",
            "### Root reviews",
            "",
            "| Source | Locator | Status | Selected body | Treatment | Verdict | Gold | Correct | Next stage |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        )
    )
    for row in score.rows:
        result = "yes" if row.matches_gold is True else "no" if row.matches_gold is False else None
        lines.append(
            f"| {_review_cell(Path(row.source_path).name)} | {_review_cell(row.locator_quote)} "
            f"({row.locator_start}:{row.locator_end}) | {_review_cell(row.review_status)} | "
            f"{_review_cell(row.selected_body_id)} | {_review_cell(row.treatment)} | "
            f"{_review_cell(row.verdict)} | {_review_cell(row.gold_identity)} | "
            f"{_review_cell(result)} | {_review_cell(row.next_stage)} |"
        )
    if not score.rows:
        lines.append("| — | — | — | — | — | — | — | — | — |")
    for row in score.rows:
        lines.extend(
            (
                "",
                f"#### {_review_cell(Path(row.source_path).name)} · {_review_cell(row.citation_id)}",
                "",
                "| Detail | Value |",
                "| --- | --- |",
                f"| Selected source | {_review_cell(row.selected_source)} |",
                f"| Evidence index | {_review_cell(row.selected_evidence_index)} |",
                f"| Selected body ID | {_review_cell(row.selected_body_id)} |",
                f"| Selected body URL | {_review_cell(row.selected_body_url)} |",
                f"| Excerpt offset in body | {_review_cell(row.selected_source_offset)} |",
                f"| Grounded citation | {_review_cell(row.grounded_citation_quote)} |",
                f"| Grounded citation span in excerpt | {_review_span_cell(row.grounded_citation_span)} |",
                f"| Grounded context | {_review_cell(row.grounded_context_quote)} |",
                f"| Grounded context span in excerpt | {_review_span_cell(row.grounded_context_span)} |",
                f"| Judgment reason | {_review_cell(row.reason)} |",
                f"| Review failure | {_review_cell(row.failure_reason)} |",
            )
        )
        if row.comparisons is not None:
            lines.extend(
                (
                    "",
                    "| Printed field | Filing | Third party | Comparison | Reason |",
                    "| --- | --- | --- | --- | --- |",
                )
            )
            for field in ("locator", *FIELDS):
                comparison = row.comparisons[field]
                lines.append(
                    f"| {field} | {_review_cell(row.filing[field] if row.filing else None)} | "
                    f"{_review_cell(row.third_party[field] if row.third_party else None)} | "
                    f"{_review_cell(comparison['result'])} | {_review_cell(comparison['reason'])} |"
                )
    return "\n".join(lines) + "\n"


def render_validate_roots(
    score: WorkflowScore,
    *,
    set_name: str | None = None,
    include_stages: bool = True,
) -> str:
    if tuple(stage.stage for stage in score.stages) != STAGES or tuple(score.fields) != FIELDS:
        raise ValueError("Validate-roots score has missing or out-of-order stages or fields")
    sections = [f"# Validate-roots evaluation{f': {set_name}' if set_name else ''}"]
    if score.checkpoint == LOCATOR_BODY_REVIEW:
        sections.append(
            f"Checkpoint: {LOCATOR_BODY_REVIEW} completed. Field identity judgments are scored through "
            f"{WORKFLOW_STAGES[-1]}; the body review's printed citation comparisons have no corresponding "
            "field identity gold. Its overall identity verdict is scored separately."
        )
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
    if score.body_review is not None:
        sections.append(render_locator_body_review(score.body_review).rstrip())
    sections.append(
        "\n".join(
            (
                (
                    "## Root identity after locator-body review"
                    if score.body_review is not None
                    else "## Lookup-derived root identity"
                ),
                "",
                (
                    "A locator-body verdict takes precedence over a lookup-derived verdict. "
                    "Undetermined case names remain in the recall denominator."
                    if score.body_review is not None
                    else "A selected lookup record is required. Undetermined case names are excluded from precision and remain in the recall denominator."
                ),
                "",
                "| Precision | Recall | Undetermined |",
                "| ---: | ---: | ---: |",
                f"| {_field_cell(FieldScore(score.identity.correct, score.identity.predicted, score.identity.gold), recall=False)} | "
                f"{_field_cell(FieldScore(score.identity.correct, score.identity.predicted, score.identity.gold), recall=True)} | "
                f"{score.identity.undetermined} |",
            )
        )
    )
    return "\n\n".join(sections) + "\n"
