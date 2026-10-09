"""Score field and identity judgments through the root-lookup workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from evaluations.annotations import load_annotations
from evaluations.score_types import (
    FieldScore,
    Precision,
    SubstageScore,
    group_substage_records,
    render_stage_sections,
    substage_heading,
)
from mellea_lrc.model import Document, FullDocketCitation, FullReporterCitation
from mellea_lrc.model.citations.body_evidence import BodySource
from mellea_lrc.model.citations.docket_lookup import DocketLookupReviewDecision
from mellea_lrc.model.citations.full import FullCitation
from mellea_lrc.model.citations.judgments import IdentityVerdict, MatchResult
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactAmbiguityOutcome,
    ReporterExactLookupOutcome,
)
from mellea_lrc.validation.body_search.intended_case_courtlistener_opinion_retrieval import (
    SUBSTAGE as INTENDED_CASE_COURTLISTENER_OPINION_RETRIEVAL,
)
from mellea_lrc.validation.body_search.intended_case_courtlistener_recap_retrieval import (
    SUBSTAGE as INTENDED_CASE_COURTLISTENER_RECAP_RETRIEVAL,
)
from mellea_lrc.validation.body_search.intended_case_govinfo_opinion_retrieval import (
    SUBSTAGE as INTENDED_CASE_GOVINFO_OPINION_RETRIEVAL,
)
from mellea_lrc.validation.body_search.locator_body_courtlistener_opinion_retrieval import (
    SUBSTAGE as LOCATOR_BODY_COURTLISTENER_OPINION_RETRIEVAL,
)
from mellea_lrc.validation.body_search.locator_body_courtlistener_recap_retrieval import (
    SUBSTAGE as LOCATOR_BODY_COURTLISTENER_RECAP_RETRIEVAL,
)
from mellea_lrc.validation.body_search.locator_body_govinfo_opinion_retrieval import (
    SUBSTAGE as LOCATOR_BODY_GOVINFO_OPINION_RETRIEVAL,
)
from mellea_lrc.validation.docket_root_lookup_courtlistener_llm_review import (
    SUBSTAGE as DOCKET_ROOT_LOOKUP_COURTLISTENER_LLM_REVIEW,
)
from mellea_lrc.validation.docket_root_lookup_courtlistener_retrieval import (
    SUBSTAGE as DOCKET_ROOT_LOOKUP_COURTLISTENER_RETRIEVAL,
)
from mellea_lrc.validation.docket_root_lookup_govinfo_llm_review import (
    SUBSTAGE as DOCKET_ROOT_LOOKUP_GOVINFO_LLM_REVIEW,
)
from mellea_lrc.validation.docket_root_lookup_govinfo_retrieval import (
    SUBSTAGE as DOCKET_ROOT_LOOKUP_GOVINFO_RETRIEVAL,
)
from mellea_lrc.validation.fields_aggregated_identity import SUBSTAGE as FIELDS_AGGREGATED_IDENTITY
from mellea_lrc.validation.intended_case_llm_selection import (
    SUBSTAGE as INTENDED_CASE_LLM_SELECTION,
)
from mellea_lrc.validation.locator_body_llm_judgment import SUBSTAGE as LOCATOR_BODY_LLM_JUDGMENT
from mellea_lrc.validation.reporter_root_lookup_ambiguous_llm_judgment import (
    SUBSTAGE as REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM_JUDGMENT,
)
from mellea_lrc.validation.reporter_root_lookup_ambiguous_rule_judgment import (
    SUBSTAGE as REPORTER_ROOT_LOOKUP_AMBIGUOUS_RULE_JUDGMENT,
)
from mellea_lrc.validation.reporter_root_lookup_cluster_retrieval import (
    SUBSTAGE as REPORTER_ROOT_LOOKUP_CLUSTER_RETRIEVAL,
)
from mellea_lrc.validation.reporter_root_lookup_docket_retrieval import (
    SUBSTAGE as REPORTER_ROOT_LOOKUP_DOCKET_RETRIEVAL,
)
from mellea_lrc.validation.reporter_root_lookup_unique_llm_judgment import (
    SUBSTAGE as REPORTER_ROOT_LOOKUP_UNIQUE_LLM_JUDGMENT,
)
from mellea_lrc.validation.reporter_root_lookup_unique_rule_judgment import (
    SUBSTAGE as REPORTER_ROOT_LOOKUP_UNIQUE_RULE_JUDGMENT,
)

FIELDS = ("case_name", "court", "date")
GOLD_LABELS = frozenset({"agrees", "disagrees", "not_stated"})
GOLD_IDENTITIES = frozenset({"CORRECT_IDENTITY", "WRONG_IDENTITY"})
SUBSTAGES = (
    REPORTER_ROOT_LOOKUP_UNIQUE_RULE_JUDGMENT,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS_RULE_JUDGMENT,
    REPORTER_ROOT_LOOKUP_UNIQUE_LLM_JUDGMENT,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM_JUDGMENT,
    DOCKET_ROOT_LOOKUP_COURTLISTENER_LLM_REVIEW,
    DOCKET_ROOT_LOOKUP_GOVINFO_LLM_REVIEW,
)
WORKFLOW_SUBSTAGES = (
    REPORTER_ROOT_LOOKUP_CLUSTER_RETRIEVAL,
    REPORTER_ROOT_LOOKUP_DOCKET_RETRIEVAL,
    REPORTER_ROOT_LOOKUP_UNIQUE_RULE_JUDGMENT,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS_RULE_JUDGMENT,
    REPORTER_ROOT_LOOKUP_UNIQUE_LLM_JUDGMENT,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM_JUDGMENT,
    DOCKET_ROOT_LOOKUP_COURTLISTENER_RETRIEVAL,
    DOCKET_ROOT_LOOKUP_COURTLISTENER_LLM_REVIEW,
    DOCKET_ROOT_LOOKUP_GOVINFO_RETRIEVAL,
    SUBSTAGES[-1],
    FIELDS_AGGREGATED_IDENTITY,
)
LOCATOR_BODY_SUBSTAGES = (
    LOCATOR_BODY_COURTLISTENER_OPINION_RETRIEVAL,
    LOCATOR_BODY_COURTLISTENER_RECAP_RETRIEVAL,
    LOCATOR_BODY_GOVINFO_OPINION_RETRIEVAL,
    LOCATOR_BODY_LLM_JUDGMENT,
)
INTENDED_CASE_SUBSTAGES = (
    INTENDED_CASE_COURTLISTENER_OPINION_RETRIEVAL,
    INTENDED_CASE_COURTLISTENER_RECAP_RETRIEVAL,
    INTENDED_CASE_GOVINFO_OPINION_RETRIEVAL,
    INTENDED_CASE_LLM_SELECTION,
)


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
class RetrievalScore:
    """Citation coverage among the roots actually queried by one retrieval substage."""

    substage: str
    citations_with_records: int = 0
    citations_queried: int = 0

    def __add__(self, other: RetrievalScore) -> RetrievalScore:
        if self.substage != other.substage:
            raise ValueError("Cannot combine different retrieval substages")
        return RetrievalScore(
            self.substage,
            self.citations_with_records + other.citations_with_records,
            self.citations_queried + other.citations_queried,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "substage": self.substage,
            "record_coverage": {
                "citations_with_records": self.citations_with_records,
                "citations_queried": self.citations_queried,
                "rate": (
                    self.citations_with_records / self.citations_queried if self.citations_queried else None
                ),
            },
        }


@dataclass(frozen=True)
class BodyReviewScore:
    """Identity verdicts issued by the locator-body review substage."""

    substage: str
    verdict_counts: dict[str, int]

    def __add__(self, other: BodyReviewScore) -> BodyReviewScore:
        if self.substage != other.substage:
            raise ValueError("Cannot combine different body-review substages")

        def combine(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
            return {key: left.get(key, 0) + right.get(key, 0) for key in left.keys() | right.keys()}

        return BodyReviewScore(self.substage, combine(self.verdict_counts, other.verdict_counts))

    def as_dict(self) -> dict[str, object]:
        return {
            "substage": self.substage,
            "verdict_counts": dict(sorted(self.verdict_counts.items())),
        }


@dataclass(frozen=True)
class FieldsAggregatedIdentityScore:
    """Identity verdicts persisted by the docket field-aggregation substage."""

    substage: str
    verdict_counts: dict[str, int]

    def __add__(self, other: FieldsAggregatedIdentityScore) -> FieldsAggregatedIdentityScore:
        if self.substage != other.substage:
            raise ValueError("Cannot combine different field-aggregation substages")
        return FieldsAggregatedIdentityScore(
            self.substage,
            {
                key: self.verdict_counts.get(key, 0) + other.verdict_counts.get(key, 0)
                for key in self.verdict_counts.keys() | other.verdict_counts.keys()
            },
        )

    def as_dict(self) -> dict[str, object]:
        return {"substage": self.substage, "verdict_counts": dict(sorted(self.verdict_counts.items()))}


@dataclass(frozen=True)
class WorkflowScore:
    substages: tuple[SubstageScore, ...]
    fields: dict[str, FieldScore]
    identity: IdentityScore
    checkpoint: str | None = None
    body_review: BodyReviewScore | None = None
    identity_with_partial: IdentityScore | None = None
    intended_case_outcomes: dict[str, int] | None = None
    retrieval_substages: tuple[RetrievalScore, ...] = ()
    fields_aggregated_identity: FieldsAggregatedIdentityScore | None = None
    completed_stages: tuple[str, ...] = ()

    @property
    def substage_order(self) -> tuple[str, ...]:
        """All completed atomic steps, including retrieval-only substages."""
        return (
            WORKFLOW_SUBSTAGES
            + (LOCATOR_BODY_SUBSTAGES if self.body_review is not None else ())
            + (INTENDED_CASE_SUBSTAGES if self.intended_case_outcomes is not None else ())
        )

    def __add__(self, other: WorkflowScore) -> WorkflowScore:
        if tuple(substage.substage for substage in self.substages) != tuple(
            substage.substage for substage in other.substages
        ):
            raise ValueError("Cannot combine different validation workflows")
        if self.fields.keys() != other.fields.keys():
            raise ValueError("Cannot combine different validation fields")
        if self.checkpoint != other.checkpoint:
            raise ValueError("Cannot combine different validation checkpoints")
        if (self.body_review is None) != (other.body_review is None):
            raise ValueError("Cannot combine workflows with different body-review details")
        if (self.identity_with_partial is None) != (other.identity_with_partial is None):
            raise ValueError("Cannot combine workflows with different inclusive identity scores")
        if (self.intended_case_outcomes is None) != (other.intended_case_outcomes is None):
            raise ValueError("Cannot combine workflows with different intended-case review outcomes")
        if (self.fields_aggregated_identity is None) != (other.fields_aggregated_identity is None):
            raise ValueError("Cannot combine workflows with different field-aggregation details")
        if tuple(item.substage for item in self.retrieval_substages) != tuple(
            item.substage for item in other.retrieval_substages
        ):
            raise ValueError("Cannot combine workflows with different retrieval substages")
        return WorkflowScore(
            tuple(left + right for left, right in zip(self.substages, other.substages, strict=True)),
            {field: score + other.fields[field] for field, score in self.fields.items()},
            self.identity + other.identity,
            self.checkpoint,
            (
                self.body_review + other.body_review
                if self.body_review is not None and other.body_review is not None
                else None
            ),
            (
                self.identity_with_partial + other.identity_with_partial
                if self.identity_with_partial is not None and other.identity_with_partial is not None
                else None
            ),
            (
                {
                    key: self.intended_case_outcomes.get(key, 0) + other.intended_case_outcomes.get(key, 0)
                    for key in self.intended_case_outcomes.keys() | other.intended_case_outcomes.keys()
                }
                if self.intended_case_outcomes is not None and other.intended_case_outcomes is not None
                else None
            ),
            tuple(
                left + right
                for left, right in zip(self.retrieval_substages, other.retrieval_substages, strict=True)
            ),
            (
                self.fields_aggregated_identity + other.fields_aggregated_identity
                if self.fields_aggregated_identity is not None
                and other.fields_aggregated_identity is not None
                else None
            ),
            tuple(stage for stage in self.completed_stages if stage in other.completed_stages),
        )

    def as_dict(self) -> dict[str, object]:
        records = [item.as_dict() for item in self.substages]
        records.extend(item.as_dict() for item in self.retrieval_substages)
        if self.body_review is not None:
            records.append(self.body_review.as_dict())
        if self.fields_aggregated_identity is not None:
            records.append(self.fields_aggregated_identity.as_dict())
        if self.intended_case_outcomes is not None:
            records.append(
                {
                    "substage": INTENDED_CASE_LLM_SELECTION,
                    "outcome_counts": dict(sorted(self.intended_case_outcomes.items())),
                }
            )
        result = {
            "workflow": "validate_roots",
            "stages": group_substage_records("validate_roots", records, self.completed_stages),
            "fields": {field: score.as_dict() for field, score in self.fields.items()},
            "identity": self.identity.as_dict(),
        }
        if self.checkpoint is not None:
            result["checkpoint"] = self.checkpoint
        if self.identity_with_partial is not None:
            result["identity_with_partial"] = self.identity_with_partial.as_dict()
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
    roots: list[_GoldRoot] = []
    spans: set[tuple[str, int, int]] = set()
    for row in load_annotations(document):
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


def _selected_candidate(root: FullReporterCitation, substage: str) -> int | None:
    if substage == REPORTER_ROOT_LOOKUP_UNIQUE_RULE_JUDGMENT:
        lookup = root.reporter_exact_lookup
        return 0 if lookup is not None and lookup.outcome is ReporterExactLookupOutcome.UNIQUE else None
    if substage == REPORTER_ROOT_LOOKUP_AMBIGUOUS_RULE_JUDGMENT:
        resolution = root.reporter_exact_ambiguity_resolution
        return (
            resolution.selected_candidate_index
            if resolution is not None
            and resolution.outcome is ReporterExactAmbiguityOutcome.UNIQUE_RULE_MATCH
            else None
        )
    if substage == REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM_JUDGMENT:
        review = root.reporter_ambiguous_review
        return review.decision.selected_candidate_index if review is not None and review.decision else None
    if substage == REPORTER_ROOT_LOOKUP_UNIQUE_LLM_JUDGMENT:
        review = root.reporter_unique_review
        return 0 if review is not None and review.decision is not None else None
    raise ValueError(f"Unknown validation substage: {substage}")


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
    if not any(
        node.id == review.node_id and node.substage == DOCKET_ROOT_LOOKUP_COURTLISTENER_LLM_REVIEW
        for node in root.nodes
    ):
        return None
    return review.decision


def _selected_govinfo_docket_review(root: FullDocketCitation) -> DocketLookupReviewDecision | None:
    review = root.govinfo_docket_review
    lookup = root.govinfo_docket_lookup
    if review is None or review.decision is None or review.decision.selected_candidate_index is None:
        return None
    selected = review.decision.selected_candidate_index
    if lookup is None:
        raise ValueError("GovInfo docket review selected a candidate without a saved lookup")
    if selected not in lookup.shortlisted_candidate_indices:
        raise ValueError(
            f"GovInfo docket review selected candidate {selected} outside the saved shortlist "
            f"{lookup.shortlisted_candidate_indices}"
        )
    if not any(
        node.id == lookup.node_id and node.substage == DOCKET_ROOT_LOOKUP_GOVINFO_RETRIEVAL
        for node in root.nodes
    ) or not any(
        node.id == review.node_id and node.substage == DOCKET_ROOT_LOOKUP_GOVINFO_LLM_REVIEW
        for node in root.nodes
    ):
        return None
    return review.decision


def score_docket_root_lookup_courtlistener_llm_review(document: Document) -> SubstageScore:
    checkpoint = document.get_substage(DOCKET_ROOT_LOOKUP_COURTLISTENER_LLM_REVIEW)
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
    return SubstageScore(
        DOCKET_ROOT_LOOKUP_COURTLISTENER_LLM_REVIEW,
        {field: Precision(*counts[field]) for field in FIELDS},
    )


def score_docket_root_lookup_govinfo_llm_review(document: Document) -> SubstageScore:
    checkpoint = document.get_substage(DOCKET_ROOT_LOOKUP_GOVINFO_LLM_REVIEW)
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
    return SubstageScore(
        DOCKET_ROOT_LOOKUP_GOVINFO_LLM_REVIEW,
        {field: Precision(*counts[field]) for field in FIELDS},
    )


def _stage_judgments(
    document: Document,
    substage: str,
    candidate: Callable[[FullReporterCitation], int | None],
) -> SubstageScore:
    checkpoint = document.get_substage(substage)
    gold = _gold_roots(checkpoint)
    aligned = _align(checkpoint.roots, gold)
    counts = {field: [0, 0] for field in FIELDS}
    for prediction_index, root in enumerate(checkpoint.roots):
        if not isinstance(root, FullReporterCitation):
            continue
        node_ids = {node.id for node in root.nodes if node.substage == substage}
        if not node_ids or (candidate_index := candidate(root)) is None:
            continue
        gold_root = gold[aligned[prediction_index]] if prediction_index in aligned else None
        for field in FIELDS:
            decisions = [
                judgment
                for judgment in getattr(root, f"{field}_judgments")
                if judgment.node_id in node_ids and judgment.candidate_index == candidate_index
            ]
            # The gold labels describe the intended citation. Comparisons to
            # rejected candidates are useful rule-substage evidence, but have no
            # candidate-specific gold and are not predictions about that case.
            if len(decisions) > 1:
                raise ValueError("A substage produced duplicate selected-candidate field judgments")
            if not decisions:
                continue
            counts[field][1] += 1
            label = _label(decisions[0].result, source_present=decisions[0].reading_index is not None)
            counts[field][0] += int(gold_root is not None and label == gold_root.labels[field])
    return SubstageScore(substage, {field: Precision(*counts[field]) for field in FIELDS})


def score_reporter_root_lookup_unique_rule_judgment(document: Document) -> SubstageScore:
    return _stage_judgments(
        document,
        REPORTER_ROOT_LOOKUP_UNIQUE_RULE_JUDGMENT,
        lambda root: _selected_candidate(root, REPORTER_ROOT_LOOKUP_UNIQUE_RULE_JUDGMENT),
    )


def score_reporter_root_lookup_ambiguous_rule_judgment(document: Document) -> SubstageScore:
    return _stage_judgments(
        document,
        REPORTER_ROOT_LOOKUP_AMBIGUOUS_RULE_JUDGMENT,
        lambda root: _selected_candidate(root, REPORTER_ROOT_LOOKUP_AMBIGUOUS_RULE_JUDGMENT),
    )


def score_reporter_root_lookup_ambiguous_llm_judgment(document: Document) -> SubstageScore:
    return _stage_judgments(
        document,
        REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM_JUDGMENT,
        lambda root: _selected_candidate(root, REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM_JUDGMENT),
    )


def score_reporter_root_lookup_unique_llm_judgment(document: Document) -> SubstageScore:
    return _stage_judgments(
        document,
        REPORTER_ROOT_LOOKUP_UNIQUE_LLM_JUDGMENT,
        lambda root: _selected_candidate(root, REPORTER_ROOT_LOOKUP_UNIQUE_LLM_JUDGMENT),
    )


SUBSTAGE_SCORERS: tuple[tuple[str, Callable[[Document], SubstageScore]], ...] = (
    (REPORTER_ROOT_LOOKUP_UNIQUE_RULE_JUDGMENT, score_reporter_root_lookup_unique_rule_judgment),
    (REPORTER_ROOT_LOOKUP_AMBIGUOUS_RULE_JUDGMENT, score_reporter_root_lookup_ambiguous_rule_judgment),
    (REPORTER_ROOT_LOOKUP_UNIQUE_LLM_JUDGMENT, score_reporter_root_lookup_unique_llm_judgment),
    (REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM_JUDGMENT, score_reporter_root_lookup_ambiguous_llm_judgment),
    (DOCKET_ROOT_LOOKUP_COURTLISTENER_LLM_REVIEW, score_docket_root_lookup_courtlistener_llm_review),
    (DOCKET_ROOT_LOOKUP_GOVINFO_LLM_REVIEW, score_docket_root_lookup_govinfo_llm_review),
)


def _stage_node_ids(root: FullCitation, substage: str) -> set[str]:
    return {node.id for node in root.nodes if node.substage == substage}


def score_reporter_root_lookup_cluster_retrieval(document: Document) -> RetrievalScore:
    """Count queried reporter citations with at least one saved cluster."""
    checkpoint = document.get_substage(REPORTER_ROOT_LOOKUP_CLUSTER_RETRIEVAL)
    queried = with_records = 0
    for root in checkpoint.roots:
        if not isinstance(root, FullReporterCitation):
            continue
        lookup = root.reporter_exact_lookup
        if lookup is None or lookup.node_id not in _stage_node_ids(
            root, REPORTER_ROOT_LOOKUP_CLUSTER_RETRIEVAL
        ):
            continue
        if lookup.query is not None:
            queried += 1
            with_records += int(lookup.response is not None and bool(lookup.response.clusters))
    return RetrievalScore(REPORTER_ROOT_LOOKUP_CLUSTER_RETRIEVAL, with_records, queried)


def score_reporter_root_lookup_docket_retrieval(document: Document) -> RetrievalScore:
    """Count queried reporter citations with at least one saved docket."""
    checkpoint = document.get_substage(REPORTER_ROOT_LOOKUP_DOCKET_RETRIEVAL)
    queried = with_records = 0
    for root in checkpoint.roots:
        if not isinstance(root, FullReporterCitation):
            continue
        node_ids = _stage_node_ids(root, REPORTER_ROOT_LOOKUP_DOCKET_RETRIEVAL)
        dockets = [
            *(
                (root.reporter_exact_docket,)
                if root.reporter_exact_docket is not None and root.reporter_exact_docket.node_id in node_ids
                else ()
            ),
            *(item for item in root.reporter_exact_candidate_dockets if item.node_id in node_ids),
        ]
        if dockets:
            queried += 1
            with_records += int(any(item.response is not None for item in dockets))
    return RetrievalScore(REPORTER_ROOT_LOOKUP_DOCKET_RETRIEVAL, with_records, queried)


def _score_docket_retrieval(document: Document, substage: str) -> RetrievalScore:
    checkpoint = document.get_substage(substage)
    queried = with_records = 0
    for root in checkpoint.roots:
        if not isinstance(root, FullDocketCitation):
            continue
        lookup = (
            root.docket_lookup
            if substage == DOCKET_ROOT_LOOKUP_COURTLISTENER_RETRIEVAL
            else root.govinfo_docket_lookup
        )
        if lookup is None or lookup.node_id not in _stage_node_ids(root, substage):
            continue
        if lookup.attempts:
            queried += 1
            with_records += int(bool(lookup.candidates))
    return RetrievalScore(substage, with_records, queried)


def score_docket_root_lookup_courtlistener_retrieval(document: Document) -> RetrievalScore:
    return _score_docket_retrieval(document, DOCKET_ROOT_LOOKUP_COURTLISTENER_RETRIEVAL)


def score_docket_root_lookup_govinfo_retrieval(document: Document) -> RetrievalScore:
    return _score_docket_retrieval(document, DOCKET_ROOT_LOOKUP_GOVINFO_RETRIEVAL)


_BODY_RETRIEVAL_SOURCES = {
    LOCATOR_BODY_COURTLISTENER_OPINION_RETRIEVAL: BodySource.COURTLISTENER_OPINION,
    LOCATOR_BODY_COURTLISTENER_RECAP_RETRIEVAL: BodySource.COURTLISTENER_RECAP,
    LOCATOR_BODY_GOVINFO_OPINION_RETRIEVAL: BodySource.GOVINFO_OPINION,
    INTENDED_CASE_COURTLISTENER_OPINION_RETRIEVAL: BodySource.COURTLISTENER_OPINION,
    INTENDED_CASE_COURTLISTENER_RECAP_RETRIEVAL: BodySource.COURTLISTENER_RECAP,
    INTENDED_CASE_GOVINFO_OPINION_RETRIEVAL: BodySource.GOVINFO_OPINION,
}


def _score_body_retrieval(document: Document, substage: str) -> RetrievalScore:
    checkpoint = document.get_substage(substage)
    queried = with_records = 0
    source = _BODY_RETRIEVAL_SOURCES[substage]
    field_stage = substage in INTENDED_CASE_SUBSTAGES
    for root in checkpoint.roots:
        node_ids = _stage_node_ids(root, substage)
        searches = root.field_body_searches if field_stage else root.body_searches
        selected = [search for search in searches if search.node_id in node_ids and search.source is source]
        if len(selected) > 1:
            raise ValueError(f"Multiple saved retrieval results for {substage} on {root.id}")
        if selected and selected[0].attempts:
            queried += 1
            with_records += int(bool(selected[0].evidence))
    return RetrievalScore(substage, with_records, queried)


def score_locator_body_courtlistener_opinion_retrieval(document: Document) -> RetrievalScore:
    return _score_body_retrieval(document, LOCATOR_BODY_COURTLISTENER_OPINION_RETRIEVAL)


def score_locator_body_courtlistener_recap_retrieval(document: Document) -> RetrievalScore:
    return _score_body_retrieval(document, LOCATOR_BODY_COURTLISTENER_RECAP_RETRIEVAL)


def score_locator_body_govinfo_opinion_retrieval(document: Document) -> RetrievalScore:
    return _score_body_retrieval(document, LOCATOR_BODY_GOVINFO_OPINION_RETRIEVAL)


def score_intended_case_courtlistener_opinion_retrieval(document: Document) -> RetrievalScore:
    return _score_body_retrieval(document, INTENDED_CASE_COURTLISTENER_OPINION_RETRIEVAL)


def score_intended_case_courtlistener_recap_retrieval(document: Document) -> RetrievalScore:
    return _score_body_retrieval(document, INTENDED_CASE_COURTLISTENER_RECAP_RETRIEVAL)


def score_intended_case_govinfo_opinion_retrieval(document: Document) -> RetrievalScore:
    return _score_body_retrieval(document, INTENDED_CASE_GOVINFO_OPINION_RETRIEVAL)


RETRIEVAL_SUBSTAGE_SCORERS: tuple[tuple[str, Callable[[Document], RetrievalScore]], ...] = (
    (REPORTER_ROOT_LOOKUP_CLUSTER_RETRIEVAL, score_reporter_root_lookup_cluster_retrieval),
    (REPORTER_ROOT_LOOKUP_DOCKET_RETRIEVAL, score_reporter_root_lookup_docket_retrieval),
    (DOCKET_ROOT_LOOKUP_COURTLISTENER_RETRIEVAL, score_docket_root_lookup_courtlistener_retrieval),
    (DOCKET_ROOT_LOOKUP_GOVINFO_RETRIEVAL, score_docket_root_lookup_govinfo_retrieval),
    (LOCATOR_BODY_COURTLISTENER_OPINION_RETRIEVAL, score_locator_body_courtlistener_opinion_retrieval),
    (LOCATOR_BODY_COURTLISTENER_RECAP_RETRIEVAL, score_locator_body_courtlistener_recap_retrieval),
    (LOCATOR_BODY_GOVINFO_OPINION_RETRIEVAL, score_locator_body_govinfo_opinion_retrieval),
    (INTENDED_CASE_COURTLISTENER_OPINION_RETRIEVAL, score_intended_case_courtlistener_opinion_retrieval),
    (INTENDED_CASE_COURTLISTENER_RECAP_RETRIEVAL, score_intended_case_courtlistener_recap_retrieval),
    (INTENDED_CASE_GOVINFO_OPINION_RETRIEVAL, score_intended_case_govinfo_opinion_retrieval),
)
RETRIEVAL_SUBSTAGES = tuple(substage for substage, _ in RETRIEVAL_SUBSTAGE_SCORERS)


def _final_reporter_field_label(
    root: FullReporterCitation, field: str, substage_runs: tuple[str, ...]
) -> str | None:
    selected_stages = [
        (substage, candidate_index)
        for substage in substage_runs
        if substage in SUBSTAGES[:-2]
        if (candidate_index := _selected_candidate(root, substage)) is not None
    ]
    if not selected_stages:
        return None
    for substage, candidate_index in reversed(selected_stages):
        node_ids = {node.id for node in root.nodes if node.substage == substage}
        selected = [
            judgment
            for judgment in getattr(root, f"{field}_judgments")
            if judgment.node_id in node_ids and judgment.candidate_index == candidate_index
        ]
        if len(selected) > 1:
            raise ValueError("A substage produced duplicate selected-candidate field judgments")
        if selected:
            return _label(selected[0].result, source_present=selected[0].reading_index is not None)
    return None


def _final_reporter_identity_verdict(
    root: FullReporterCitation, substage_runs: tuple[str, ...]
) -> str | None:
    """Use an identity verdict actually saved on a selected lookup node."""
    for substage in reversed(substage_runs):
        if substage not in SUBSTAGES[:-2]:
            continue
        if _selected_candidate(root, substage) is None:
            continue
        node_ids = {node.id for node in root.nodes if node.substage == substage}
        judgments = [judgment for judgment in root.identity_judgments if judgment.node_id in node_ids]
        if len(judgments) > 1:
            raise ValueError("A lookup substage produced duplicate identity judgments")
        if judgments:
            return judgments[0].verdict.name.upper()
    return None


def _final_docket_field_label(root: FullDocketCitation, field: str) -> str | None:
    decision = _selected_govinfo_docket_review(root) or _selected_docket_review(root)
    if decision is None:
        return None
    return _label(getattr(decision, field).result, source_present=bool(getattr(root, field)))


def _final_field_label(root: FullCitation, field: str, substage_runs: tuple[str, ...]) -> str | None:
    if isinstance(root, FullDocketCitation):
        return _final_docket_field_label(root, field)
    if isinstance(root, FullReporterCitation):
        return _final_reporter_field_label(root, field, substage_runs)
    raise ValueError(f"Unsupported validation root kind: {type(root).__name__}")


def _identity_value(labels: dict[str, str]) -> str | None:
    """Resolve a complete selected lookup; incomplete field outcomes stay uncomputed."""
    allowed = {"agrees", "disagrees", "unavailable", "not_stated"}
    if any(label not in allowed for label in labels.values()) or not set(labels) <= set(FIELDS):
        raise ValueError("Identity contains an invalid field judgment")
    if set(labels) != set(FIELDS):
        return None
    if "disagrees" in labels.values():
        return "WRONG_IDENTITY"
    if labels["case_name"] in {"unavailable", "not_stated"}:
        return "UNDETERMINED"
    return "CORRECT_IDENTITY"


def _fields_aggregated_identity_verdict(root: FullDocketCitation) -> IdentityVerdict | None:
    """Read the persisted aggregation verdict; never synthesize missing judgments."""
    node_ids = _stage_node_ids(root, FIELDS_AGGREGATED_IDENTITY)
    judgments = [judgment for judgment in root.identity_judgments if judgment.node_id in node_ids]
    selected = _selected_govinfo_docket_review(root) or _selected_docket_review(root)
    if selected is None:
        if node_ids or judgments:
            raise ValueError("Field-aggregated identity requires a selected docket review")
        return None
    if len(node_ids) != 1 or len(judgments) != 1:
        raise ValueError("Selected docket review must issue exactly one field-aggregated identity judgment")
    verdict = judgments[0].verdict
    if verdict is IdentityVerdict.PARTIALLY_CORROBORATED:
        raise ValueError("Field-aggregated identity cannot issue a partially corroborated verdict")
    return verdict


def score_fields_aggregated_identity(document: Document) -> FieldsAggregatedIdentityScore:
    """Count only identity verdicts persisted by field aggregation."""
    checkpoint = document.get_substage(FIELDS_AGGREGATED_IDENTITY)
    verdict_counts: dict[str, int] = {}
    for root in checkpoint.roots:
        if not isinstance(root, FullDocketCitation):
            continue
        verdict = _fields_aggregated_identity_verdict(root)
        if verdict is not None:
            verdict_counts[verdict.value] = verdict_counts.get(verdict.value, 0) + 1
    return FieldsAggregatedIdentityScore(FIELDS_AGGREGATED_IDENTITY, verdict_counts)


def _body_verdict(root: FullCitation) -> IdentityVerdict | None:
    if not root.body_reviews:
        return None
    review = root.body_reviews[-1]
    if not any(
        node.id == review.node_id and node.substage == LOCATOR_BODY_LLM_JUDGMENT for node in root.nodes
    ):
        raise ValueError("Body review does not reference its substage node")
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


def score_locator_body_llm_judgment(document: Document) -> BodyReviewScore:
    """Count locator-body identity verdicts; routes are not issued judgments."""
    checkpoint = document.get_substage(LOCATOR_BODY_LLM_JUDGMENT)
    verdict_counts: dict[str, int] = {}
    for root in checkpoint.roots:
        stage_nodes = [node for node in root.nodes if node.substage == LOCATOR_BODY_LLM_JUDGMENT]
        if not stage_nodes:
            if root.body_reviews:
                raise ValueError("Body review does not reference its substage node")
            continue
        reviews = [
            review for review in root.body_reviews if review.node_id in {node.id for node in stage_nodes}
        ]
        if len(stage_nodes) != 1 or len(reviews) != 1:
            raise ValueError("Locator-body review must record exactly one review for each processed root")
        verdict = _body_verdict(root)
        route = _body_route(root, reviews[0].node_id)
        if (verdict is None or verdict.name not in GOLD_IDENTITIES) != (route is not None):
            raise ValueError("Unresolved body reviews must route; decisive verdicts must clear the route")
        if verdict is not None:
            verdict_counts[verdict.value] = verdict_counts.get(verdict.value, 0) + 1
    return BodyReviewScore(LOCATOR_BODY_LLM_JUDGMENT, verdict_counts)


def score_intended_case_llm_selection(document: Document) -> dict[str, int]:
    """Count intended-case candidate outcomes without judging their accuracy."""
    checkpoint = document.get_substage(INTENDED_CASE_LLM_SELECTION)
    outcomes: dict[str, int] = {}
    for root in checkpoint.roots:
        stage_nodes = {node.id for node in root.nodes if node.substage == INTENDED_CASE_LLM_SELECTION}
        if not stage_nodes:
            continue
        reviews = [review for review in root.intended_case_reviews if review.node_id in stage_nodes]
        if len(stage_nodes) != 1 or len(reviews) != 1:
            raise ValueError("Intended-case selection needs one review for each processed root")
        if any(judgment.node_id in stage_nodes for judgment in root.identity_judgments):
            raise ValueError("Intended-case review cannot issue an identity judgment")
        review = reviews[0]
        if review.failure_reason is not None:
            outcome = "review_failure"
        elif review.decision is not None and review.decision.source is None:
            outcome = "declined"
        elif review.decision is not None and review.decision.confidence is not None:
            outcome = f"selected_{review.decision.confidence.value}"
        else:
            raise ValueError("Intended-case review has no valid outcome")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    return outcomes


def score_validate_roots(document: Document) -> WorkflowScore:
    """Score completed validation judgments and every annotated root's final fields."""
    if any(substage not in document.substage_runs for substage in WORKFLOW_SUBSTAGES):
        missing = [substage for substage in WORKFLOW_SUBSTAGES if substage not in document.substage_runs]
        raise ValueError(f"Incomplete validate_roots workflow; missing substages: {', '.join(missing)}")
    if (
        tuple(substage for substage in document.substage_runs if substage in WORKFLOW_SUBSTAGES)
        != WORKFLOW_SUBSTAGES
    ):
        raise ValueError("Validate-roots workflow substages are out of execution order")
    body_stages = set(LOCATOR_BODY_SUBSTAGES).intersection(document.substage_runs)
    if body_stages and body_stages != set(LOCATOR_BODY_SUBSTAGES):
        missing = [substage for substage in LOCATOR_BODY_SUBSTAGES if substage not in document.substage_runs]
        raise ValueError(
            f"Incomplete locator-body substages in validate_roots; missing substages: {', '.join(missing)}"
        )
    field_body_stages = set(INTENDED_CASE_SUBSTAGES).intersection(document.substage_runs)
    if field_body_stages and field_body_stages != set(INTENDED_CASE_SUBSTAGES):
        missing = [substage for substage in INTENDED_CASE_SUBSTAGES if substage not in document.substage_runs]
        raise ValueError(
            f"Incomplete intended-case substages in validate_roots; missing substages: {', '.join(missing)}"
        )
    if field_body_stages and not body_stages:
        raise ValueError("Intended-case substages require completed locator-body substages")
    final_stage = LOCATOR_BODY_LLM_JUDGMENT if body_stages else FIELDS_AGGREGATED_IDENTITY
    final = document.get_substage(final_stage)
    stage_scores = tuple(score(final) for _, score in SUBSTAGE_SCORERS)
    gold = _gold_roots(final)
    # Body review can correct extracted readings, but it does not judge whether
    # those fields identify the same case as an external validation candidate.
    judged = final.get_substage(DOCKET_ROOT_LOOKUP_GOVINFO_LLM_REVIEW)
    aggregated = final.get_substage(FIELDS_AGGREGATED_IDENTITY)
    aggregated_roots = {root.id: root for root in aggregated.roots if isinstance(root, FullDocketCitation)}
    fields_aggregated_identity = score_fields_aggregated_identity(aggregated)
    aligned = _align(judged.roots, gold)
    counts = {field: [0, 0] for field in FIELDS}
    identity_correct = identity_predicted = undetermined = 0
    lookup_verdicts: dict[str, str] = {}
    for prediction_index, root in enumerate(judged.roots):
        if isinstance(root, FullReporterCitation) and not any(
            node.substage in SUBSTAGES for node in root.nodes
        ):
            continue
        gold_root = gold[aligned[prediction_index]] if prediction_index in aligned else None
        outcomes: dict[str, str] = {}
        for field in FIELDS:
            outcome = _final_field_label(root, field, judged.substage_runs)
            if outcome is None:
                continue
            outcomes[field] = outcome
            counts[field][1] += 1
            counts[field][0] += int(gold_root is not None and outcome == gold_root.labels[field])
        if isinstance(root, FullDocketCitation):
            saved_verdict = _fields_aggregated_identity_verdict(aggregated_roots[root.id])
            verdict = saved_verdict.name if saved_verdict is not None else None
        else:
            if not outcomes:
                continue
            verdict = _identity_value(outcomes)
            if verdict is None:
                verdict = _final_reporter_identity_verdict(root, judged.substage_runs)
        if verdict is None:
            continue
        lookup_verdicts[root.id] = verdict
        if verdict == "UNDETERMINED":
            undetermined += 1
        else:
            identity_predicted += 1
            identity_correct += int(gold_root is not None and verdict == gold_root.identity)
    identity = IdentityScore(identity_correct, identity_predicted, len(gold), undetermined)
    body_review = None
    identity_with_partial = None
    if body_stages:
        body_review = score_locator_body_llm_judgment(final)
        final_aligned = _align(final.roots, gold)
        identity_correct = identity_predicted = undetermined = 0
        inclusive_correct = inclusive_predicted = inclusive_undetermined = 0
        for prediction_index, root in enumerate(final.roots):
            body_verdict = _body_verdict(root)
            if body_verdict is IdentityVerdict.PARTIALLY_CORROBORATED:
                # This qualified verdict supports the case, but not the specific decision.
                # The broader metric treats it as a positive against binary identity gold.
                verdict = "UNDETERMINED"
                inclusive_verdict = "CORRECT_IDENTITY"
            elif body_verdict is IdentityVerdict.UNDETERMINED:
                # A nondecisive body judgment retracts an earlier full-citation admission.
                verdict = inclusive_verdict = "UNDETERMINED"
            elif body_verdict is not None and body_verdict.name in GOLD_IDENTITIES:
                verdict = inclusive_verdict = body_verdict.name
            else:
                verdict = inclusive_verdict = lookup_verdicts.get(root.id)
            if verdict is None:
                continue
            if verdict == "UNDETERMINED":
                undetermined += 1
            else:
                identity_predicted += 1
                identity_correct += int(
                    prediction_index in final_aligned
                    and verdict == gold[final_aligned[prediction_index]].identity
                )
            if inclusive_verdict == "UNDETERMINED":
                inclusive_undetermined += 1
            else:
                inclusive_predicted += 1
                inclusive_correct += int(
                    prediction_index in final_aligned
                    and inclusive_verdict == gold[final_aligned[prediction_index]].identity
                )
        identity = IdentityScore(identity_correct, identity_predicted, len(gold), undetermined)
        identity_with_partial = IdentityScore(
            inclusive_correct, inclusive_predicted, len(gold), inclusive_undetermined
        )
    return WorkflowScore(
        stage_scores,
        {field: FieldScore(counts[field][0], counts[field][1], len(gold)) for field in FIELDS},
        identity,
        INTENDED_CASE_LLM_SELECTION
        if field_body_stages
        else LOCATOR_BODY_LLM_JUDGMENT
        if body_stages
        else FIELDS_AGGREGATED_IDENTITY,
        body_review,
        identity_with_partial,
        score_intended_case_llm_selection(document) if field_body_stages else None,
        tuple(
            scorer(document)
            for substage, scorer in RETRIEVAL_SUBSTAGE_SCORERS
            if substage in WORKFLOW_SUBSTAGES
            or (body_stages and substage in LOCATOR_BODY_SUBSTAGES)
            or (field_body_stages and substage in INTENDED_CASE_SUBSTAGES)
        ),
        fields_aggregated_identity,
        tuple(stage for stage in document.stage_runs if stage.startswith("validate_roots.")),
    )


def _precision_cell(score: Precision) -> str:
    return "—" if score.total == 0 else f"{score.correct}/{score.total} ({score.correct / score.total:.1%})"


def _field_cell(score: FieldScore, *, recall: bool) -> str:
    denominator = score.gold if recall else score.predicted
    return "—" if denominator == 0 else f"{score.correct}/{denominator} ({score.correct / denominator:.1%})"


def _render_substage(score: SubstageScore, expected: str) -> str:
    if score.substage != expected or tuple(score.metrics) != FIELDS:
        raise ValueError(f"Expected {expected} field score")
    lines = [
        substage_heading(expected),
        "",
        "| Field | Precision |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {field} | {_precision_cell(score.metrics[field])} |" for field in FIELDS)
    return "\n".join(lines)


def render_retrieval_substage(score: RetrievalScore) -> str:
    if score.substage not in RETRIEVAL_SUBSTAGES:
        raise ValueError(f"Unknown retrieval substage: {score.substage}")
    fraction = f"{score.citations_with_records}/{score.citations_queried}"
    if score.citations_queried:
        fraction += f" ({score.citations_with_records / score.citations_queried:.1%})"
    return (
        f"{substage_heading(score.substage)}\n\n"
        "| Metric | Coverage |\n"
        "| --- | ---: |\n"
        f"| Citations with records / citations queried | {fraction} |\n"
    )


def render_reporter_root_lookup_unique_rule_judgment(score: SubstageScore) -> str:
    return _render_substage(score, REPORTER_ROOT_LOOKUP_UNIQUE_RULE_JUDGMENT) + "\n"


def render_docket_root_lookup_courtlistener_llm_review(score: SubstageScore) -> str:
    return _render_substage(score, DOCKET_ROOT_LOOKUP_COURTLISTENER_LLM_REVIEW) + "\n"


def render_docket_root_lookup_govinfo_llm_review(score: SubstageScore) -> str:
    return _render_substage(score, DOCKET_ROOT_LOOKUP_GOVINFO_LLM_REVIEW) + "\n"


def render_reporter_root_lookup_ambiguous_rule_judgment(score: SubstageScore) -> str:
    return _render_substage(score, REPORTER_ROOT_LOOKUP_AMBIGUOUS_RULE_JUDGMENT) + "\n"


def render_reporter_root_lookup_ambiguous_llm_judgment(score: SubstageScore) -> str:
    return _render_substage(score, REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM_JUDGMENT) + "\n"


def render_reporter_root_lookup_unique_llm_judgment(score: SubstageScore) -> str:
    return _render_substage(score, REPORTER_ROOT_LOOKUP_UNIQUE_LLM_JUDGMENT) + "\n"


def render_fields_aggregated_identity(score: FieldsAggregatedIdentityScore) -> str:
    """Render the issued aggregation verdicts without additional accuracy denominators."""
    if score.substage != FIELDS_AGGREGATED_IDENTITY:
        raise ValueError(f"Expected {FIELDS_AGGREGATED_IDENTITY} aggregation score")
    lines = [
        f"{substage_heading(FIELDS_AGGREGATED_IDENTITY)}",
        "",
        "| Issued verdict | Count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {verdict} | {count} |" for verdict, count in sorted(score.verdict_counts.items()))
    if not score.verdict_counts:
        lines.append("| — | 0 |")
    return "\n".join(lines) + "\n"


def render_locator_body_llm_judgment(score: BodyReviewScore) -> str:
    """Render only the identity verdicts issued by locator-body review."""
    if score.substage != LOCATOR_BODY_LLM_JUDGMENT:
        raise ValueError(f"Expected {LOCATOR_BODY_LLM_JUDGMENT} review score")
    lines = [
        f"{substage_heading(LOCATOR_BODY_LLM_JUDGMENT)}",
        "",
        "| Issued verdict | Count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {verdict} | {count} |" for verdict, count in sorted(score.verdict_counts.items()))
    if not score.verdict_counts:
        lines.append("| — | 0 |")
    return "\n".join(lines) + "\n"


def render_intended_case_llm_selection(outcomes: dict[str, int]) -> str:
    """Render candidate counts without implying an accuracy measure."""
    lines = [
        f"{substage_heading(INTENDED_CASE_LLM_SELECTION)}",
        "",
        "Candidate outcomes only; the annotations do not label intended-case candidates.",
        "",
        "| Outcome | Count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {outcome} | {count} |" for outcome, count in sorted(outcomes.items()))
    if not outcomes:
        lines.append("| — | 0 |")
    return "\n".join(lines) + "\n"


def _identity_cell(canonical: IdentityScore, inclusive: IdentityScore | None, *, recall: bool) -> str:
    value = _field_cell(FieldScore(canonical.correct, canonical.predicted, canonical.gold), recall=recall)
    if inclusive is None:
        return value
    inclusive_denominator = inclusive.gold if recall else inclusive.predicted
    broader = (
        f"{inclusive.correct}/{inclusive_denominator} = {inclusive.correct / inclusive_denominator:.1%}"
        if inclusive_denominator
        else "—"
    )
    return f"{value} (including partial: {broader})"


def render_validate_roots(
    score: WorkflowScore,
    *,
    set_name: str | None = None,
    include_stages: bool = True,
) -> str:
    if tuple(substage.substage for substage in score.substages) != SUBSTAGES or tuple(score.fields) != FIELDS:
        raise ValueError("Validate-roots score has missing or out-of-order substages or fields")
    if (score.body_review is None) != (score.identity_with_partial is None):
        raise ValueError("Body-review workflows require both canonical and inclusive identity scores")
    if score.fields_aggregated_identity is None:
        raise ValueError("Validate-roots workflow requires a field-aggregation score")
    substage_order = score.substage_order
    if tuple(item.substage for item in score.retrieval_substages) != tuple(
        substage for substage in substage_order if substage in RETRIEVAL_SUBSTAGES
    ):
        raise ValueError("Validate-roots score has missing or out-of-order retrieval scores")
    sections = [f"# Validate-roots evaluation{f': {set_name}' if set_name else ''}"]
    if score.checkpoint in {LOCATOR_BODY_LLM_JUDGMENT, INTENDED_CASE_LLM_SELECTION}:
        sections.append(
            f"Checkpoint: {score.checkpoint} completed. Field identity judgments are scored through "
            f"{DOCKET_ROOT_LOOKUP_GOVINFO_LLM_REVIEW}; the body review's printed citation comparisons have no corresponding "
            "field identity gold. The locator-body review's overall identity verdict is scored separately. "
            "Later intended-case candidates do not change that identity score."
        )
    lines = [
        "## Root field judgments after docket lookup",
        "",
        "| Field | Precision | Recall |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {field} | {_field_cell(score.fields[field], recall=False)} | "
        f"{_field_cell(score.fields[field], recall=True)} |"
        for field in FIELDS
    )
    field_summary = "\n".join(lines)
    if include_stages:
        details = []
        scored_stages = {substage.substage: substage for substage in score.substages}
        retrieval_substages = {substage.substage: substage for substage in score.retrieval_substages}
        for substage in substage_order:
            if substage in scored_stages:
                details.append((substage, _render_substage(scored_stages[substage], substage)))
            elif substage in retrieval_substages:
                details.append((substage, render_retrieval_substage(retrieval_substages[substage])))
            elif substage == FIELDS_AGGREGATED_IDENTITY:
                details.append(
                    (substage, render_fields_aggregated_identity(score.fields_aggregated_identity))
                )
            elif substage == LOCATOR_BODY_LLM_JUDGMENT:
                if score.body_review is None:
                    raise ValueError("Missing locator-body review score")
                details.append((substage, render_locator_body_llm_judgment(score.body_review)))
            elif substage == INTENDED_CASE_LLM_SELECTION:
                if score.intended_case_outcomes is None:
                    raise ValueError("Missing intended-case review score")
                details.append((substage, render_intended_case_llm_selection(score.intended_case_outcomes)))
            else:
                raise ValueError(f"Missing substage detail: {substage}")
        sections.extend(render_stage_sections("validate_roots", details, score.completed_stages))
    sections.append(field_summary)
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
                    "Parenthetical figures include partially_corroborated as a positive prediction; "
                    "it counts as correct only against CORRECT_IDENTITY gold. "
                    "Undetermined roots remain in the recall denominator."
                    if score.body_review is not None
                    else "A selected lookup record is required. Undetermined identities are excluded from precision and remain in the recall denominator."
                ),
                "",
                "| Precision | Recall | Undetermined |",
                "| ---: | ---: | ---: |",
                f"| {_identity_cell(score.identity, score.identity_with_partial, recall=False)} | "
                f"{_identity_cell(score.identity, score.identity_with_partial, recall=True)} | "
                f"{score.identity.undetermined} |",
            )
        )
    )
    return "\n\n".join(sections) + "\n"
