"""Evaluate pinpoint stages from their cumulative saved Documents."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from evaluations.annotations import citation_annotations
from evaluations.grow_leaves import _attribution_rows, _gold, _gold_key, _key, _span
from mellea_lrc.model import Document, FullReporterCitation
from mellea_lrc.model.citations import latest
from mellea_lrc.model.citations.fields.pin_cite import PinCiteTarget
from mellea_lrc.model.citations.judgments import IdentityVerdict
from mellea_lrc.model.citations.reporter_pinpoint import (
    OpinionReviewScope,
    OpinionSupportResult,
    PinpointPageAssessment,
    ReporterPinpointVerdict,
)
from mellea_lrc.validation.reporter_root_opinion_retrieval import (
    STAGE,
    selected_reporter_root_cluster,
)

_SET = "primary"
PAGE_INDEX_STAGE = "40_reporter_root_opinion_page_index"
PAGE_RESOLUTION_STAGE = "41_reporter_citation_page_resolution"
OPINION_SELECTION_STAGE = "42_reporter_citation_opinion_review"
PROPOSITION_STAGE = "43_reporter_citation_propositions"
PINPOINT_EVIDENCE_STAGE = "44_reporter_citation_pinpoint_evidence"
PAGE_SUPPORT_STAGE = "45_reporter_citation_pinpoint_page_review"
FULL_OPINION_STAGE = "46_reporter_citation_full_opinion_review"
JUDGMENT_STAGE = "47_reporter_citation_pinpoint_judgment"
LATER_STAGES = (
    PAGE_INDEX_STAGE,
    PAGE_RESOLUTION_STAGE,
    OPINION_SELECTION_STAGE,
    PROPOSITION_STAGE,
    PINPOINT_EVIDENCE_STAGE,
    PAGE_SUPPORT_STAGE,
    FULL_OPINION_STAGE,
    JUDGMENT_STAGE,
)
GOLD_LABELS = frozenset({"CORRECT_PINCITE", "WRONG_PINCITE"})
DATASET_GROUPS = ("reporter_roots", "reporter_leaves", "docket_roots", "docket_leaves")
GROUPS = (*DATASET_GROUPS, "total")
DATASET_COUNTS = (
    "total_pincites",
    "settled",
    "CORRECT_PINCITE",
    "WRONG_PINCITE",
    "SKIPPED_UNSETTLED",
    "SKIPPED_TOA",
    "SKIPPED_IDENTITY_WRONG",
    "missing_pin_label",
)


@dataclass(frozen=True)
class DatasetInventory:
    """Native annotation counts, independent of every pipeline decision."""

    all_annotations: dict[str, int]
    under_gold_correct_identity_roots: dict[str, int]
    settled_by_family: dict[str, dict[str, int]]
    settled_under_gold_correct_identity_roots: dict[str, dict[str, int]]
    known_page_locations: dict[str, int]

    def __add__(self, other: DatasetInventory) -> DatasetInventory:
        return DatasetInventory(
            {key: self.all_annotations[key] + other.all_annotations[key] for key in DATASET_COUNTS},
            {
                key: self.under_gold_correct_identity_roots[key]
                + other.under_gold_correct_identity_roots[key]
                for key in DATASET_COUNTS
            },
            {
                group: {
                    label: self.settled_by_family[group][label] + other.settled_by_family[group][label]
                    for label in GOLD_LABELS
                }
                for group in DATASET_GROUPS
            },
            {
                group: {
                    label: self.settled_under_gold_correct_identity_roots[group][label]
                    + other.settled_under_gold_correct_identity_roots[group][label]
                    for label in GOLD_LABELS
                }
                for group in DATASET_GROUPS
            },
            {group: self.known_page_locations[group] + other.known_page_locations[group] for group in GROUPS},
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": "Native citation annotations only; each leaf uses its root's gold identity",
            "all_annotations": self.all_annotations,
            "under_gold_correct_identity_roots": self.under_gold_correct_identity_roots,
            "settled_judgments": {
                group: {
                    "all_annotations": self.settled_by_family[group],
                    "under_gold_correct_identity_roots": self.settled_under_gold_correct_identity_roots[
                        group
                    ],
                }
                for group in DATASET_GROUPS
            },
            "known_page_locations_under_gold_correct_identity_roots": self.known_page_locations,
        }


def _dataset_inventory(document: Document) -> DatasetInventory:
    rows = {row["id"]: row for row in citation_annotations(document)}
    all_counts: Counter[str] = Counter()
    correct_identity_counts: Counter[str] = Counter()
    settled = {group: Counter() for group in DATASET_GROUPS}
    correct_identity_settled = {group: Counter() for group in DATASET_GROUPS}
    known_page_locations: Counter[str] = Counter()
    for row in rows.values():
        finding = row.get("validation", {}).get("pincite", {})
        pin = row.get("pin_cite", {})
        if not (
            finding.get("label")
            or _span(pin) is not None
            or pin.get("source", {}).get("kind") in {"quoted", "inferred"}
        ):
            continue
        root = rows[row["root_id"]]
        correct_identity = root.get("validation", {}).get("identity", {}).get("label") == "CORRECT_IDENTITY"
        label = finding.get("label")
        status = f"SKIPPED_{finding['skip_type']}" if label == "SKIPPED" else label or "missing_pin_label"
        if status not in DATASET_COUNTS:
            raise ValueError(f"Unknown native pin cite status: {status}")
        for counts in (all_counts, correct_identity_counts) if correct_identity else (all_counts,):
            counts["total_pincites"] += 1
            counts[status] += 1
            counts["settled"] += label in GOLD_LABELS
        if label in GOLD_LABELS:
            family = "docket" if root["kind"] == "DocketCitation" else "reporter"
            group = f"{family}_{'roots' if row['is_root'] else 'leaves'}"
            settled[group][label] += 1
            correct_identity_settled[group][label] += correct_identity
            if correct_identity and _gold_page_location(row) is not None:
                known_page_locations[group] += 1
                known_page_locations["total"] += 1
    return DatasetInventory(
        {key: all_counts[key] for key in DATASET_COUNTS},
        {key: correct_identity_counts[key] for key in DATASET_COUNTS},
        {group: {label: settled[group][label] for label in sorted(GOLD_LABELS)} for group in DATASET_GROUPS},
        {
            group: {label: correct_identity_settled[group][label] for label in sorted(GOLD_LABELS)}
            for group in DATASET_GROUPS
        },
        {group: known_page_locations[group] for group in GROUPS},
    )


@dataclass(frozen=True)
class OpinionRetrievalScore:
    stage: str = STAGE
    reporter_roots_opinion_retrievals: int = 0
    reporter_roots_correct_identity: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.reporter_roots_opinion_retrievals <= self.reporter_roots_correct_identity:
            raise ValueError("Opinion retrievals must belong to the eligible reporter-root cohort")

    def __add__(self, other: OpinionRetrievalScore) -> OpinionRetrievalScore:
        if self.stage != other.stage:
            raise ValueError("Cannot combine different opinion retrieval stages")
        return type(self)(
            stage=self.stage,
            reporter_roots_opinion_retrievals=(
                self.reporter_roots_opinion_retrievals + other.reporter_roots_opinion_retrievals
            ),
            reporter_roots_correct_identity=(
                self.reporter_roots_correct_identity + other.reporter_roots_correct_identity
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "reporter_roots_opinion_retrievals": self.reporter_roots_opinion_retrievals,
            "reporter_roots_correct_identity": self.reporter_roots_correct_identity,
            "ratio": self.reporter_roots_opinion_retrievals / self.reporter_roots_correct_identity
            if self.reporter_roots_correct_identity
            else None,
        }


def score_reporter_root_opinion_retrieval(document: Document) -> OpinionRetrievalScore:
    checkpoint = document.get_stage(STAGE)
    eligible = retrieved = 0
    for citation in checkpoint.roots:
        if not isinstance(citation, FullReporterCitation):
            continue
        if (
            not citation.identity_judgments
            or citation.identity_judgments[-1].verdict is not IdentityVerdict.CORRECT_IDENTITY
            or (
                citation.reporter_root_opinion_source is None
                and selected_reporter_root_cluster(citation) is None
            )
        ):
            continue
        eligible += 1
        result = citation.reporter_root_opinion_retrieval
        retrieved += result is not None and any(opinion.text_field is not None for opinion in result.opinions)
    return OpinionRetrievalScore(
        reporter_roots_opinion_retrievals=retrieved,
        reporter_roots_correct_identity=eligible,
    )


@dataclass(frozen=True)
class PinpointScore:
    """Definitive judgment accuracy and recall over native settled pin cites."""

    correct: int = 0
    predicted: int = 0
    gold: int = 0
    undetermined: int = 0
    missing_opinions: int = 0

    def __add__(self, other: PinpointScore) -> PinpointScore:
        return PinpointScore(
            self.correct + other.correct,
            self.predicted + other.predicted,
            self.gold + other.gold,
            self.undetermined + other.undetermined,
            self.missing_opinions + other.missing_opinions,
        )

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "correct": self.correct,
            "predicted": self.predicted,
            "gold": self.gold,
            "wrong": self.predicted - self.correct,
            "undetermined": self.undetermined,
            "missing_judgments": self.gold - self.predicted - self.undetermined,
            "missing_opinions": self.missing_opinions,
            "precision": self.correct / self.predicted if self.predicted else None,
            "recall": self.correct / self.gold if self.gold else None,
        }


@dataclass(frozen=True)
class PagePrecision:
    """Explicit Boolean page assertions with native page gold, independent of support."""

    correct: int = 0
    predicted: int = 0
    unlocated: int = 0
    unscored: int = 0

    def __add__(self, other: PagePrecision) -> PagePrecision:
        return type(self)(
            self.correct + other.correct,
            self.predicted + other.predicted,
            self.unlocated + other.unlocated,
            self.unscored + other.unscored,
        )

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "correct": self.correct,
            "predicted": self.predicted,
            "unlocated": self.unlocated,
            "unscored": self.unscored,
            "precision": self.correct / self.predicted if self.predicted else None,
        }


@dataclass(frozen=True)
class FoundPageAgreement(PagePrecision):
    """Recovered targets scored only where native positive references settle them."""

    def as_dict(self) -> dict[str, int | float | None]:
        result = super().as_dict()
        result["reference_agreement"] = result.pop("precision")
        return result


@dataclass(frozen=True)
class PinpointStageScore:
    stage: str
    counts: dict[str, int]
    judgments: dict[str, PinpointScore] | None = None
    unscored_definitive: int = 0
    page_precision: dict[str, PagePrecision] | None = None
    found_page_precision: dict[str, FoundPageAgreement] | None = None

    def __add__(self, other: PinpointStageScore) -> PinpointStageScore:
        if self.stage != other.stage or (self.judgments is None) != (other.judgments is None):
            raise ValueError("Cannot combine different pinpoint stages")
        if (self.page_precision is None) != (other.page_precision is None):
            raise ValueError("Cannot combine different page evaluation boundaries")
        if (self.found_page_precision is None) != (other.found_page_precision is None):
            raise ValueError("Cannot combine different found-page evaluation boundaries")
        return PinpointStageScore(
            self.stage,
            {
                key: self.counts.get(key, 0) + other.counts.get(key, 0)
                for key in self.counts.keys() | other.counts.keys()
            },
            {group: self.judgments[group] + other.judgments[group] for group in GROUPS}
            if self.judgments is not None and other.judgments is not None
            else None,
            self.unscored_definitive + other.unscored_definitive,
            {group: self.page_precision[group] + other.page_precision[group] for group in GROUPS}
            if self.page_precision is not None and other.page_precision is not None
            else None,
            {group: self.found_page_precision[group] + other.found_page_precision[group] for group in GROUPS}
            if self.found_page_precision is not None and other.found_page_precision is not None
            else None,
        )

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"stage": self.stage, "counts": self.counts}
        if self.judgments is not None:
            result["judgments"] = {
                group: {
                    "correct": score.correct,
                    "predicted": score.predicted,
                    "undetermined": score.undetermined,
                    "precision": score.correct / score.predicted if score.predicted else None,
                }
                for group, score in self.judgments.items()
            }
            result["unscored_definitive"] = self.unscored_definitive
        if self.page_precision is not None:
            result["page_precision"] = {
                group: score.as_dict() for group, score in self.page_precision.items()
            }
        if self.found_page_precision is not None:
            result["found_page_precision"] = {
                group: score.as_dict() for group, score in self.found_page_precision.items()
            }
        return result


@dataclass(frozen=True)
class WorkflowScore:
    stages: tuple[OpinionRetrievalScore | PinpointStageScore, ...]
    pinpoint: dict[str, PinpointScore] | None = None
    dataset: DatasetInventory | None = None
    page_precision: dict[str, PagePrecision] | None = None
    found_page_precision: dict[str, FoundPageAgreement] | None = None

    def __add__(self, other: WorkflowScore) -> WorkflowScore:
        if (self.pinpoint is None) != (other.pinpoint is None):
            raise ValueError("Cannot combine different pinpoint workflow boundaries")
        if (self.dataset is None) != (other.dataset is None):
            raise ValueError("Cannot combine workflows with different annotation availability")
        if (self.page_precision is None) != (other.page_precision is None):
            raise ValueError("Cannot combine workflows with different page evaluation boundaries")
        if (self.found_page_precision is None) != (other.found_page_precision is None):
            raise ValueError("Cannot combine workflows with different found-page evaluation boundaries")
        return type(self)(
            tuple(left + right for left, right in zip(self.stages, other.stages, strict=True)),
            {group: self.pinpoint[group] + other.pinpoint[group] for group in GROUPS}
            if self.pinpoint is not None and other.pinpoint is not None
            else None,
            self.dataset + other.dataset if self.dataset is not None and other.dataset is not None else None,
            {group: self.page_precision[group] + other.page_precision[group] for group in GROUPS}
            if self.page_precision is not None and other.page_precision is not None
            else None,
            {group: self.found_page_precision[group] + other.found_page_precision[group] for group in GROUPS}
            if self.found_page_precision is not None and other.found_page_precision is not None
            else None,
        )

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"dataset": self.dataset.as_dict()} if self.dataset is not None else {}
        result["stages"] = [stage.as_dict() for stage in self.stages]
        if self.pinpoint is not None:
            result["gold_cohort"] = {
                "scope": "Native settled pin cites under gold CORRECT_IDENTITY roots, including reporter and docket roots and leaves",
                "labels": sorted(GOLD_LABELS),
                **{group: self.pinpoint[group].gold for group in GROUPS},
            }
            result["pinpoint"] = {group: score.as_dict() for group, score in self.pinpoint.items()}
        if self.page_precision is not None:
            result["page_precision"] = {
                group: score.as_dict() for group, score in self.page_precision.items()
            }
        if self.found_page_precision is not None:
            result["found_page_precision"] = {
                group: score.as_dict() for group, score in self.found_page_precision.items()
            }
        return result


def _entries(citation: Any, field: str, stage: str) -> tuple[Any, ...]:
    nodes = {node.id for node in citation.nodes if node.stage == stage}
    return tuple(entry for entry in getattr(citation, field) if entry.node_id in nodes)


def _review_verdict(review: Any) -> ReporterPinpointVerdict:
    if review.decision is None:
        return ReporterPinpointVerdict.UNDETERMINED
    result = review.decision.result
    if result is OpinionSupportResult.SUPPORTED:
        return ReporterPinpointVerdict.CORRECT_PINCITE
    if review.scope is OpinionReviewScope.FULL_OPINION and result in {
        OpinionSupportResult.CONTRADICTED,
        OpinionSupportResult.NOT_FOUND,
    }:
        return ReporterPinpointVerdict.WRONG_PINCITE
    return ReporterPinpointVerdict.UNDETERMINED


def _settled_correct_identity_gold(document: Document) -> tuple[dict, dict]:
    gold = _gold(document)
    rows = {row["id"]: row for row in gold.values()}
    correct_identity_roots = {
        row["id"]
        for row in rows.values()
        if row["is_root"] and row.get("validation", {}).get("identity", {}).get("label") == "CORRECT_IDENTITY"
    }
    cohort = {
        row["id"]: row
        for row in rows.values()
        if row["root_id"] in correct_identity_roots
        and row.get("validation", {}).get("pincite", {}).get("label") in GOLD_LABELS
    }
    return gold, cohort


def _score_judgments(
    document: Document, stage: str, *, cumulative: bool = False
) -> tuple[dict[str, PinpointScore], int]:
    """Do not narrow gold by pipeline admission, retrieval, provider, or page selection."""
    gold, cohort = _settled_correct_identity_gold(document)
    rows = {row["id"]: row for row in gold.values()}
    roots = {_key(root): root for root in document.roots if isinstance(root, FullReporterCitation)}
    counts = {
        group: {"correct": 0, "predicted": 0, "gold": 0, "undetermined": 0, "missing_opinions": 0}
        for group in GROUPS
    }
    for row in cohort.values():
        family = "docket" if rows[row["root_id"]]["kind"] == "DocketCitation" else "reporter"
        group = f"{family}_{'roots' if row['is_root'] else 'leaves'}"
        root = roots.get(_gold_key(rows[row["root_id"]]))
        retrieval = root.reporter_root_opinion_retrieval if root is not None else None
        missing = retrieval is None or not any(
            opinion.text_field is not None for opinion in retrieval.opinions
        )
        for category in (group, "total"):
            counts[category]["gold"] += 1
            counts[category]["missing_opinions"] += missing
    unscored = 0
    for citation, row in _attribution_rows(list(document.citations), gold):
        field = "reporter_pinpoint_judgments" if stage == JUDGMENT_STAGE else "reporter_support_reviews"
        if cumulative and stage != JUDGMENT_STAGE:
            nodes = {
                node.id for node in citation.nodes if node.stage in {PAGE_SUPPORT_STAGE, FULL_OPINION_STAGE}
            }
            entries = tuple(entry for entry in getattr(citation, field) if entry.node_id in nodes)
        else:
            entries = _entries(citation, field, stage)
        if not entries:
            continue
        verdict = entries[-1].verdict if stage == JUDGMENT_STAGE else _review_verdict(entries[-1])
        if row is None or row["id"] not in cohort:
            unscored += verdict is not ReporterPinpointVerdict.UNDETERMINED
            continue
        family = "docket" if rows[row["root_id"]]["kind"] == "DocketCitation" else "reporter"
        group = f"{family}_{'roots' if row['is_root'] else 'leaves'}"
        predicted_root = next((root for root in document.roots if root.id == latest(citation.root_id)), None)
        attachment_agrees = predicted_root is not None and _key(predicted_root) == _gold_key(
            rows[row["root_id"]]
        )
        for category in (group, "total"):
            if verdict is ReporterPinpointVerdict.UNDETERMINED:
                counts[category]["undetermined"] += 1
            else:
                counts[category]["predicted"] += 1
                counts[category]["correct"] += (
                    attachment_agrees and verdict.value == row["validation"]["pincite"]["label"]
                )
    return {group: PinpointScore(**values) for group, values in counts.items()}, unscored


def _gold_page_location(row: dict) -> bool | None:
    finding = row.get("validation", {}).get("pincite", {})
    value = finding.get("correct_page")
    return value if type(value) is bool else None


def _page_predictions(
    document: Document, stage: str, gold: dict, *, cumulative: bool
) -> Iterator[tuple[Any, dict | None, PinpointPageAssessment]]:
    for citation, row in _attribution_rows(list(document.citations), gold):
        if stage == JUDGMENT_STAGE:
            entries = _entries(citation, "reporter_pinpoint_judgments", stage)
            if not entries:
                continue
            assessment = entries[-1]
        else:
            entries = (
                tuple(
                    entry
                    for entry in citation.reporter_support_reviews
                    if any(
                        node.id == entry.node_id and node.stage in {PAGE_SUPPORT_STAGE, FULL_OPINION_STAGE}
                        for node in citation.nodes
                    )
                )
                if cumulative
                else _entries(citation, "reporter_support_reviews", stage)
            )
            if not entries or entries[-1].decision is None:
                continue
            assessment = entries[-1].decision
        yield citation, row, assessment


def _page_categories(row: dict | None, cohort: dict, rows: dict) -> tuple[str, ...]:
    if row is None or row["id"] not in cohort:
        return ("total",)
    family = "docket" if rows[row["root_id"]]["kind"] == "DocketCitation" else "reporter"
    return (f"{family}_{'roots' if row['is_root'] else 'leaves'}", "total")


def _page_attachment_agrees(citation: Any, row: dict, roots: dict, rows: dict) -> bool:
    root = roots.get(latest(citation.root_id))
    return root is not None and _key(root) == _gold_key(rows[row["root_id"]])


def _score_page_locations(
    document: Document, stage: str, *, cumulative: bool = False
) -> dict[str, PagePrecision]:
    gold, cohort = _settled_correct_identity_gold(document)
    rows = {row["id"]: row for row in gold.values()}
    roots = {root.id: root for root in document.roots}
    counts = {group: Counter() for group in GROUPS}
    for citation, row, assessment in _page_predictions(document, stage, gold, cumulative=cumulative):
        categories = _page_categories(row, cohort, rows)
        expected = _gold_page_location(row) if row is not None and row["id"] in cohort else None
        if expected is None:
            for category in categories:
                counts[category]["unscored"] += 1
            continue
        if assessment.correct_page is None:
            for category in categories:
                counts[category]["unlocated"] += 1
            continue
        attachment_agrees = _page_attachment_agrees(citation, row, roots, rows)
        for category in categories:
            counts[category]["predicted"] += 1
            counts[category]["correct"] += attachment_agrees and assessment.correct_page is expected
    return {group: PagePrecision(**values) for group, values in counts.items()}


def _target_covered(target: PinCiteTarget, references: tuple[PinCiteTarget, ...]) -> bool:
    """Compare positive typed intervals without assuming unlisted locations false."""
    cursor = target.first
    for reference in sorted(references, key=lambda item: (item.first, item.last)):
        if reference.kind is not target.kind or reference.footnote != target.footnote:
            continue
        if reference.last < cursor:
            continue
        if reference.first > cursor:
            return False
        cursor = reference.last + 1
        if cursor > target.last:
            return True
    return False


def _score_found_page_locations(
    document: Document, stage: str, *, cumulative: bool = False
) -> dict[str, FoundPageAgreement]:
    gold, cohort = _settled_correct_identity_gold(document)
    rows = {row["id"]: row for row in gold.values()}
    roots = {root.id: root for root in document.roots}
    counts = {group: Counter() for group in GROUPS}
    for citation, row, assessment in _page_predictions(document, stage, gold, cumulative=cumulative):
        categories = _page_categories(row, cohort, rows)
        references = (
            tuple(
                PinCiteTarget.model_validate(item)
                for item in row["validation"]["pincite"].get("found_pages", [])
            )
            if row is not None and row["id"] in cohort
            else ()
        )
        written = (
            tuple(citation.pin_cite[-1].get_normalized())
            if citation.pin_cite and citation.pin_cite[-1].normalizable
            else ()
        )
        contradicted_target = (
            row is not None
            and row["id"] in cohort
            and _gold_page_location(row) is False
            and any(_target_covered(target, written) for target in assessment.found_pages)
        )
        confirmed = bool(references) and all(
            _target_covered(target, references) for target in assessment.found_pages
        )
        if not references and not contradicted_target:
            status = "unscored"
        elif not assessment.found_pages:
            status = "unlocated"
        elif confirmed or contradicted_target:
            status = "predicted"
        else:
            status = "unscored"
        for category in categories:
            counts[category][status] += 1
            if status == "predicted":
                counts[category]["correct"] += (
                    confirmed
                    and not contradicted_target
                    and _page_attachment_agrees(citation, row, roots, rows)
                )
    return {group: FoundPageAgreement(**values) for group, values in counts.items()}


def score_reporter_root_opinion_page_index(document: Document) -> PinpointStageScore:
    checkpoint = document.get_stage(PAGE_INDEX_STAGE)
    counts: Counter[str] = Counter()
    for citation in checkpoint.citations:
        index = getattr(citation, "reporter_root_opinion_page_index", None)
        if index is not None and any(
            node.id == index.node_id and node.stage == PAGE_INDEX_STAGE for node in citation.nodes
        ):
            counts["roots_indexed"] += 1
            counts["opinions_indexed"] += len(index.opinions)
    return PinpointStageScore(PAGE_INDEX_STAGE, dict(counts))


def score_reporter_citation_page_resolution(document: Document) -> PinpointStageScore:
    checkpoint = document.get_stage(PAGE_RESOLUTION_STAGE)
    counts = Counter(
        entry.outcome.value
        for citation in checkpoint.citations
        for entry in _entries(citation, "reporter_page_resolutions", PAGE_RESOLUTION_STAGE)
    )
    return PinpointStageScore(PAGE_RESOLUTION_STAGE, dict(counts))


def score_reporter_citation_opinion_review(document: Document) -> PinpointStageScore:
    checkpoint = document.get_stage(OPINION_SELECTION_STAGE)
    counts = Counter(
        "decisions" if entry.decision is not None else "failures"
        for citation in checkpoint.citations
        for entry in _entries(citation, "reporter_opinion_reviews", OPINION_SELECTION_STAGE)
    )
    return PinpointStageScore(OPINION_SELECTION_STAGE, dict(counts))


def score_reporter_citation_propositions(document: Document) -> PinpointStageScore:
    checkpoint = document.get_stage(PROPOSITION_STAGE)
    counts: Counter[str] = Counter()
    for citation in checkpoint.citations:
        for entry in _entries(citation, "reporter_propositions", PROPOSITION_STAGE):
            counts["decisions" if entry.decision is not None else "failures"] += 1
            if entry.decision is not None and not entry.passages:
                counts["no_proposition"] += 1
    return PinpointStageScore(PROPOSITION_STAGE, dict(counts))


def score_reporter_citation_pinpoint_evidence(document: Document) -> PinpointStageScore:
    checkpoint = document.get_stage(PINPOINT_EVIDENCE_STAGE)
    counts = Counter(
        entry.outcome.value
        for citation in checkpoint.citations
        for entry in _entries(citation, "reporter_pinpoint_evidence", PINPOINT_EVIDENCE_STAGE)
    )
    return PinpointStageScore(PINPOINT_EVIDENCE_STAGE, dict(counts))


def score_reporter_citation_pinpoint_page_review(document: Document) -> PinpointStageScore:
    checkpoint = document.get_stage(PAGE_SUPPORT_STAGE)
    counts = Counter(
        entry.decision.result.value if entry.decision is not None else "failures"
        for citation in checkpoint.citations
        for entry in _entries(citation, "reporter_support_reviews", PAGE_SUPPORT_STAGE)
    )
    judgments, unscored = _score_judgments(checkpoint, PAGE_SUPPORT_STAGE)
    return PinpointStageScore(PAGE_SUPPORT_STAGE, dict(counts), judgments, unscored)


def score_reporter_citation_full_opinion_review(document: Document) -> PinpointStageScore:
    checkpoint = document.get_stage(FULL_OPINION_STAGE)
    counts = Counter(
        entry.decision.result.value if entry.decision is not None else "failures"
        for citation in checkpoint.citations
        for entry in _entries(citation, "reporter_support_reviews", FULL_OPINION_STAGE)
    )
    judgments, unscored = _score_judgments(checkpoint, FULL_OPINION_STAGE)
    return PinpointStageScore(
        FULL_OPINION_STAGE,
        dict(counts),
        judgments,
        unscored,
        _score_page_locations(checkpoint, FULL_OPINION_STAGE),
        _score_found_page_locations(checkpoint, FULL_OPINION_STAGE),
    )


def score_reporter_citation_pinpoint_judgment(document: Document) -> PinpointStageScore:
    checkpoint = document.get_stage(JUDGMENT_STAGE)
    counts = Counter(
        entry.verdict.value
        for citation in checkpoint.citations
        for entry in _entries(citation, "reporter_pinpoint_judgments", JUDGMENT_STAGE)
    )
    judgments, unscored = _score_judgments(checkpoint, JUDGMENT_STAGE)
    return PinpointStageScore(
        JUDGMENT_STAGE,
        dict(counts),
        judgments,
        unscored,
        _score_page_locations(checkpoint, JUDGMENT_STAGE),
        _score_found_page_locations(checkpoint, JUDGMENT_STAGE),
    )


_STAGE_SCORERS: dict[str, Callable[[Document], OpinionRetrievalScore | PinpointStageScore]] = {
    STAGE: score_reporter_root_opinion_retrieval,
    PAGE_INDEX_STAGE: score_reporter_root_opinion_page_index,
    PAGE_RESOLUTION_STAGE: score_reporter_citation_page_resolution,
    OPINION_SELECTION_STAGE: score_reporter_citation_opinion_review,
    PROPOSITION_STAGE: score_reporter_citation_propositions,
    PINPOINT_EVIDENCE_STAGE: score_reporter_citation_pinpoint_evidence,
    PAGE_SUPPORT_STAGE: score_reporter_citation_pinpoint_page_review,
    FULL_OPINION_STAGE: score_reporter_citation_full_opinion_review,
    JUDGMENT_STAGE: score_reporter_citation_pinpoint_judgment,
}


def score_validate_pincite(document: Document) -> WorkflowScore:
    stages: list[OpinionRetrievalScore | PinpointStageScore] = [_STAGE_SCORERS[STAGE](document)]
    stages.extend(_STAGE_SCORERS[stage](document) for stage in LATER_STAGES if stage in document.stage_runs)
    summary_stage = next(
        (stage for stage in reversed(LATER_STAGES[-3:]) if stage in document.stage_runs), None
    )
    summary = (
        _score_judgments(document.get_stage(summary_stage), summary_stage, cumulative=True)[0]
        if summary_stage
        else None
    )
    inventory = _dataset_inventory(document) if document.source_path is not None else None
    page_precision = (
        _score_page_locations(document.get_stage(summary_stage), summary_stage, cumulative=True)
        if summary_stage in {FULL_OPINION_STAGE, JUDGMENT_STAGE}
        else None
    )
    found_page_precision = (
        _score_found_page_locations(document.get_stage(summary_stage), summary_stage, cumulative=True)
        if summary_stage in {FULL_OPINION_STAGE, JUDGMENT_STAGE}
        else None
    )
    return WorkflowScore(tuple(stages), summary, inventory, page_precision, found_page_precision)


def render_reporter_root_opinion_retrieval(score: OpinionRetrievalScore) -> str:
    numerator = score.reporter_roots_opinion_retrievals
    denominator = score.reporter_roots_correct_identity
    percentage = f"{100 * numerator / denominator:.1f}%" if denominator else "—"
    return (
        f"## {score.stage}\n\n"
        "Scope: reporter roots with correct identity and a selected original CourtListener cluster.\n\n"
        "| Metric | Result |\n| --- | --- |\n"
        "| reporter_roots_opinion_retrievals / reporter_roots_correct_identity "
        f"| {numerator}/{denominator} ({percentage}) |\n"
    )


def _render_pinpoint_stage(score: PinpointStageScore, expected_stage: str) -> str:
    if score.stage != expected_stage:
        raise ValueError(f"Renderer needs a {expected_stage} score")
    lines = [f"## {score.stage}", "", "| Outcome | Count |", "| --- | ---: |"]
    lines.extend(f"| {name} | {count} |" for name, count in sorted(score.counts.items()))
    if score.judgments is not None:
        page_header = " Page precision |" if score.page_precision is not None else ""
        page_separator = " ---: |" if score.page_precision is not None else ""
        found_header = " Found-page agreement |" if score.found_page_precision is not None else ""
        found_separator = " ---: |" if score.found_page_precision is not None else ""
        lines.extend(
            [
                "",
                "Incremental definitive judgments against the settled gold cohort.",
                "",
                f"| Occurrences | Precision |{page_header}{found_header}",
                f"| --- | ---: |{page_separator}{found_separator}",
            ]
        )
        for group, result in score.judgments.items():
            page_cell = (
                f" {_fraction(score.page_precision[group].correct, score.page_precision[group].predicted)} |"
                if score.page_precision is not None
                else ""
            )
            found_cell = (
                f" {_fraction(score.found_page_precision[group].correct, score.found_page_precision[group].predicted)} |"
                if score.found_page_precision is not None
                else ""
            )
            lines.append(
                f"| {group} | {_fraction(result.correct, result.predicted)} |{page_cell}{found_cell}"
            )
        lines.extend(
            [
                "",
                f"Definitive judgments outside the scored cohort: {score.unscored_definitive} (unscored).",
            ]
        )
        if score.page_precision is not None:
            lines.extend(["", _render_page_precision_note(score.page_precision, score.stage)])
        if score.found_page_precision is not None:
            lines.extend(["", _render_found_page_note(score.found_page_precision)])
    return "\n".join(lines) + "\n"


def render_reporter_root_opinion_page_index(score: PinpointStageScore) -> str:
    return _render_pinpoint_stage(score, PAGE_INDEX_STAGE)


def render_reporter_citation_page_resolution(score: PinpointStageScore) -> str:
    return _render_pinpoint_stage(score, PAGE_RESOLUTION_STAGE)


def render_reporter_citation_opinion_review(score: PinpointStageScore) -> str:
    return _render_pinpoint_stage(score, OPINION_SELECTION_STAGE)


def render_reporter_citation_propositions(score: PinpointStageScore) -> str:
    return _render_pinpoint_stage(score, PROPOSITION_STAGE)


def render_reporter_citation_pinpoint_evidence(score: PinpointStageScore) -> str:
    return _render_pinpoint_stage(score, PINPOINT_EVIDENCE_STAGE)


def render_reporter_citation_pinpoint_page_review(score: PinpointStageScore) -> str:
    return _render_pinpoint_stage(score, PAGE_SUPPORT_STAGE)


def render_reporter_citation_full_opinion_review(score: PinpointStageScore) -> str:
    return _render_pinpoint_stage(score, FULL_OPINION_STAGE)


def render_reporter_citation_pinpoint_judgment(score: PinpointStageScore) -> str:
    return _render_pinpoint_stage(score, JUDGMENT_STAGE)


_STAGE_RENDERERS: dict[str, Callable[..., str]] = {
    STAGE: render_reporter_root_opinion_retrieval,
    PAGE_INDEX_STAGE: render_reporter_root_opinion_page_index,
    PAGE_RESOLUTION_STAGE: render_reporter_citation_page_resolution,
    OPINION_SELECTION_STAGE: render_reporter_citation_opinion_review,
    PROPOSITION_STAGE: render_reporter_citation_propositions,
    PINPOINT_EVIDENCE_STAGE: render_reporter_citation_pinpoint_evidence,
    PAGE_SUPPORT_STAGE: render_reporter_citation_pinpoint_page_review,
    FULL_OPINION_STAGE: render_reporter_citation_full_opinion_review,
    JUDGMENT_STAGE: render_reporter_citation_pinpoint_judgment,
}


def render_validate_pincite(score: WorkflowScore, *, set_name: str = _SET) -> str:
    reports = [_render_dataset_inventory(score.dataset)] if score.dataset is not None else []
    reports.extend(_STAGE_RENDERERS[stage.stage](stage) for stage in score.stages)
    if score.pinpoint is not None:
        page_header = " Page precision |" if score.page_precision is not None else ""
        page_separator = " ---: |" if score.page_precision is not None else ""
        found_header = " Found-page agreement |" if score.found_page_precision is not None else ""
        found_separator = " ---: |" if score.found_page_precision is not None else ""
        lines = [
            "## Pinpoint judgments",
            "",
            "The fixed gold cohort contains every native CORRECT_PINCITE or WRONG_PINCITE under a gold CORRECT_IDENTITY root. "
            "Reporter and docket roots and their leaves are included; a leaf uses its root's gold identity. "
            "SKIPPED and occurrences without a pin cite are excluded. Opinion retrieval, identity admission, "
            "and page/proposition success never narrow the recall denominator. "
            "Precision is correct definitive judgments / issued definitive judgments within this gold cohort; "
            "recall is correct definitive judgments / all settled gold. "
            "UNDETERMINED is a recall miss. Wrong verdicts receive no recall credit.",
            "",
            f"| Occurrences | Precision | Recall |{page_header}{found_header} Wrong verdicts | Undetermined | Missing judgments | Missing opinions |",
            f"| --- | ---: | ---: |{page_separator}{found_separator} ---: | ---: | ---: | ---: |",
        ]
        for group, result in score.pinpoint.items():
            cells = [
                group,
                _fraction(result.correct, result.predicted),
                _fraction(result.correct, result.gold),
            ]
            if score.page_precision is not None:
                page = score.page_precision[group]
                cells.append(_fraction(page.correct, page.predicted))
            if score.found_page_precision is not None:
                found = score.found_page_precision[group]
                cells.append(_fraction(found.correct, found.predicted))
            cells.extend(
                str(value)
                for value in (
                    result.predicted - result.correct,
                    result.undetermined,
                    result.gold - result.predicted - result.undetermined,
                    result.missing_opinions,
                )
            )
            lines.append("| " + " | ".join(cells) + " |")
        lines.extend(
            [
                "",
                "Missing-opinion counts describe coverage and may overlap undetermined or missing judgments. "
                "Support on another page or in an opinion without pagination can still be CORRECT_PINCITE. "
                "Docket pinpoint validation is not implemented, so its settled annotations remain missing judgments.",
            ]
        )
        if score.page_precision is not None:
            lines.extend(["", _render_page_precision_note(score.page_precision, score.stages[-1].stage)])
        if score.found_page_precision is not None:
            lines.extend(["", _render_found_page_note(score.found_page_precision)])
        reports.append("\n".join(lines) + "\n")
    return f"# validate_pincite — {set_name}\n\n" + "\n".join(reports)


def _render_page_precision_note(scores: dict[str, PagePrecision], stage: str) -> str:
    score = scores["total"]
    assertion = (
        "the model's correct_page assessment"
        if stage == FULL_OPINION_STAGE
        else "the final preserved correct_page assessment"
    )
    return (
        f"Page precision scores {assertion} against an explicit native Boolean, independently of content support. "
        "A content error can still have a correct written page. "
        f"{score.unlocated} assessments have known gold page correctness but no predicted Boolean; "
        f"{score.unscored} have no comparable gold placement or are outside the scored cohort. "
        "Neither enters page precision. Missing pagination or an unknown native page assessment is not a page negative. "
        "The support recall denominator is unchanged."
    )


def _render_found_page_note(scores: dict[str, FoundPageAgreement]) -> str:
    score = scores["total"]
    return (
        "Found-page agreement compares every reported numeric range, kind, and footnote with native positive page references, "
        "without matching provider opinion IDs. It is positive-reference agreement, not exhaustive factual precision: "
        "native found-page lists need not include every valid location. Unlisted or partly uncovered alternatives remain unscored; "
        "an explicit native incorrect-page finding can rule out a reported target inside the written pinpoint. "
        f"{score.unlocated} assessments have positive native references but report no found pages; "
        f"{score.unscored} lack comparable references, provide unlisted alternatives, or fall outside the scored cohort. "
        "Neither enters the agreement denominator."
    )


def _render_dataset_inventory(inventory: DatasetInventory) -> str:
    lines = [
        "## Dataset annotations",
        "",
        "Counts come only from native citation annotations. Each leaf uses its root's gold identity; "
        "pipeline extraction, retrieval, inherited pinpoints, and support judgments do not affect these counts.",
        "",
        "| Pin cite annotations | All annotations | Under gold CORRECT_IDENTITY roots |",
        "| --- | ---: | ---: |",
    ]
    for key, name in (
        ("total_pincites", "Total pin cite occurrences"),
        ("settled", "Settled subtotal"),
        ("CORRECT_PINCITE", "Settled: CORRECT_PINCITE"),
        ("WRONG_PINCITE", "Settled: WRONG_PINCITE"),
        ("SKIPPED_UNSETTLED", "SKIPPED: UNSETTLED"),
        ("SKIPPED_TOA", "SKIPPED: TOA"),
        ("SKIPPED_IDENTITY_WRONG", "SKIPPED: IDENTITY_WRONG"),
        ("missing_pin_label", "Pin cite present, no validation label"),
    ):
        lines.append(
            f"| {name} | {inventory.all_annotations[key]} | {inventory.under_gold_correct_identity_roots[key]} |"
        )
    lines.extend(
        [
            "",
            "| Settled family occurrences | All annotations | Under gold CORRECT_IDENTITY roots |",
            "| --- | ---: | ---: |",
        ]
    )
    for group in DATASET_GROUPS:
        cells = []
        for counts in (
            inventory.settled_by_family[group],
            inventory.settled_under_gold_correct_identity_roots[group],
        ):
            correct, wrong = counts["CORRECT_PINCITE"], counts["WRONG_PINCITE"]
            cells.append(f"{correct + wrong} ({correct} correct, {wrong} wrong)")
        lines.append(f"| {group} | {cells[0]} | {cells[1]} |")
    cohort_total = inventory.under_gold_correct_identity_roots["settled"]
    lines.extend(
        [
            "",
            f"The pinpoint recall cohort contains all {cohort_total} settled annotations under gold CORRECT_IDENTITY roots, "
            "including reporter and docket families. Docket pinpoint validation is not implemented, "
            "so its settled annotations remain missing judgments. "
            "SKIPPED and unlabeled pin cites do not enter correctness metrics.",
            "",
            f"Known native page correctness within that cohort: {inventory.known_page_locations['total']}. "
            "These annotations have an explicit native correct_page Boolean, independently of the content label. "
            "A null or missing correct_page is unknown and does not enter page precision.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fraction(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)" if denominator else "0/0 (—)"
