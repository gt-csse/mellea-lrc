"""Score field judgments written by the saved unique-reporter model stage.

Only explicit annotated field labels on the same root occurrence can score a
judgment. The judged reading must be the checkpoint's latest reading and must
match the annotated source reading. Identity decisions are outside this score.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluations.annotations import SETS, annotated_documents, span
from evaluations.score_identity import FIELD_LOGS, FIELD_TYPES, GOLD_FIELD_RESULT, _same_annotated_reading
from evaluations.stage_products import stage_product
from mellea_lrc.model import Document, FullReporterCitation, latest
from mellea_lrc.model.citations.judgments import MatchResult
from mellea_lrc.model.citations.reporter_lookup import ReporterExactLookupOutcome, ReporterUniqueReview
from mellea_lrc.validation.reporter_root_lookup_unique_llm import STAGE

FIELDS = ("case_name", "court", "date")


def _summary(counts: Counter[str]) -> dict[str, Any]:
    fields: dict[str, dict[str, int | float | None]] = {}
    for field in FIELDS:
        scored = counts[f"{field}_scored"]
        correct = counts[f"{field}_correct"]
        fields[field] = {
            "correct": correct,
            "scored": scored,
            "value": round(correct / scored, 4) if scored else None,
        }
    return {"field_precision": fields}


def score_document(
    document: Document, rows: tuple[dict[str, Any], ...]
) -> tuple[Counter[str], list[dict[str, Any]]]:
    """Compare the unique model stage's field products with aligned labels."""
    product = stage_product(document, STAGE)
    if product.before is None:
        raise ValueError("Unique reporter review needs a prior reporter-lookup checkpoint")
    checkpoint = product.after
    routed = {
        root.id
        for root in product.before.roots
        if isinstance(root, FullReporterCitation)
        and root.identity_judgments
        and root.identity_judgments[-1].next_stage == STAGE
    }
    roots = {root.id: root for root in checkpoint.roots if isinstance(root, FullReporterCitation)}
    if routed - roots.keys():
        raise ValueError("A routed reporter root is missing from the unique-review checkpoint")

    by_locator: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("kind") != "FullCaseCitation" or not isinstance(row.get("locator"), dict):
            continue
        locator = span(row["locator"])
        key = (locator.start, locator.end)
        if key in by_locator:
            raise ValueError("Duplicate annotated reporter locator")
        by_locator[key] = row

    reviews: dict[str, list[ReporterUniqueReview]] = defaultdict(list)
    judgments: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for item in product.records:
        if item.name not in {*FIELD_TYPES, "reporter_unique_review"}:
            continue
        if not isinstance(item.citation, FullReporterCitation) or item.citation.id not in routed:
            raise ValueError("Unique-review field or review record belongs to a nonrouted root")
        if item.name == "reporter_unique_review":
            if not isinstance(item.record, ReporterUniqueReview):
                raise ValueError("Unique-review stage wrote an unexpected review record")
            reviews[item.citation.id].append(item.record)
        else:
            if not isinstance(item.record, FIELD_TYPES[item.name]):
                raise ValueError("Unique-review stage wrote an unexpected field judgment")
            judgments[item.citation.id, item.name].append(item.record)
    if set(reviews) != routed or any(len(items) != 1 for items in reviews.values()):
        raise ValueError("Each routed reporter root needs one unique model review")

    counts: Counter[str] = Counter(documents=1, routed_roots=len(routed))
    details: list[dict[str, Any]] = []
    for root_id in sorted(routed):
        root = roots[root_id]
        lookup = root.reporter_exact_lookup
        if (
            lookup is None
            or lookup.outcome is not ReporterExactLookupOutcome.UNIQUE
            or lookup.response is None
            or len(lookup.response.clusters) != 1
        ):
            raise ValueError("Unique-review root needs one saved reporter candidate")
        candidate = lookup.response.clusters[0]
        docket = root.reporter_exact_docket.response if root.reporter_exact_docket else None
        candidate_values = {
            "case_name": candidate.case_name_full or candidate.case_name,
            "court": {
                "cluster_court_id": candidate.court_id,
                "cluster_court": candidate.court,
                "docket_court_id": docket.court_id if docket is not None else None,
                "docket_court": docket.court if docket is not None else None,
            },
            "date": candidate.date_filed,
        }
        review = reviews[root_id][0]
        annotated = by_locator.get((root.locator_span.start, root.locator_span.end))
        gold_id = annotated.get("root_id") if annotated is not None else None
        member_gold_ids = {
            row["root_id"]
            for member in checkpoint.full_locators
            if isinstance(member, FullReporterCitation)
            if latest(member.root_id) == root.id
            if (row := by_locator.get((member.locator_span.start, member.locator_span.end))) is not None
        }
        same_root_occurrence = (
            annotated is not None
            and annotated.get("is_root") is True
            and annotated.get("id") == gold_id
            and annotated.get("identifier", {}).get("kind") == "reporter"
            and len(member_gold_ids) <= 1
        )
        gold = annotated if same_root_occurrence else None
        for field in FIELDS:
            log_name = FIELD_LOGS[field]
            records = judgments.get((root_id, log_name), ())
            if len(records) > 1:
                raise ValueError(f"Unique-review stage wrote multiple {field} judgments for one root")
            if review.decision is None and records:
                raise ValueError("Failed unique review cannot write field judgments")
            if review.decision is not None and not records:
                raise ValueError(f"Successful unique review omitted its {field} judgment")
            gold_reading = gold.get(field) if gold is not None else None
            gold_label = (
                gold.get("validation", {}).get("identity", {}).get("fields", {}).get(field, {}).get("label")
                if gold is not None
                else None
            )
            if gold_label not in {*GOLD_FIELD_RESULT, "not_stated", None}:
                raise ValueError(f"Unknown annotated {field} judgment: {gold_label}")
            readings = getattr(root, field)
            newest_index = len(readings) - 1 if readings else None
            newest = readings[-1] if readings else None
            record = records[0] if records else None
            outcome: str
            if record is None:
                outcome = "review_failed"
            else:
                counts[f"{field}_judgments"] += 1
                if record.candidate_index != 0:
                    raise ValueError(f"Unique-review {field} judgment references a nonunique candidate")
                if record.reading_index is not None and record.reading_index >= len(readings):
                    raise ValueError(f"Unique-review {field} judgment has an invalid reading index")
                if gold is None:
                    outcome = (
                        "conflicting_gold_roots"
                        if len(member_gold_ids) > 1
                        else "changed_occurrence"
                        if annotated is not None
                        else "unmatched_locator"
                    )
                elif gold_label not in GOLD_FIELD_RESULT:
                    outcome = "not_stated" if gold_label == "not_stated" else "unlabeled"
                elif newest is None:
                    outcome = "missing_reading"
                elif not _same_annotated_reading(field, newest, gold_reading):
                    outcome = "misaligned_reading"
                elif record.reading_index != newest_index:
                    outcome = "stale_reading"
                elif record.result is MatchResult.UNDETERMINED:
                    outcome = "undetermined"
                    counts[f"{field}_undetermined"] += 1
                else:
                    counts[f"{field}_scored"] += 1
                    if record.result is GOLD_FIELD_RESULT[gold_label]:
                        outcome = "correct"
                        counts[f"{field}_correct"] += 1
                    else:
                        outcome = "incorrect"
                        counts[f"{field}_incorrect"] += 1
                if outcome not in {"correct", "incorrect", "undetermined"}:
                    counts[f"{field}_unscored"] += 1
            details.append(
                {
                    "product": "field_judgment" if record is not None else "field_gap",
                    "citation_id": root_id,
                    "locator_span": {"start": root.locator_span.start, "end": root.locator_span.end},
                    "gold_root_id": gold_id,
                    "field": field,
                    "gold_label": gold_label,
                    "gold_reading": gold_reading,
                    "latest_reading_index": newest_index,
                    "latest_reading": newest.model_dump(mode="json") if newest is not None else None,
                    "reading_index": record.reading_index if record is not None else None,
                    "judged_reading": (
                        readings[record.reading_index].model_dump(mode="json")
                        if record is not None and record.reading_index is not None
                        else None
                    ),
                    "candidate_index": record.candidate_index if record is not None else None,
                    "candidate_cluster_id": candidate.id,
                    "candidate_value": candidate_values[field],
                    "result": record.result.value if record is not None else None,
                    "review_failure_reason": review.failure_reason,
                    "outcome": outcome,
                }
            )
    return counts, details


def evaluate(data_root: Path, run_dir: Path, sets: tuple[str, ...]) -> dict[str, Any]:
    """Score saved checkpoints without making provider or model calls."""
    selected = tuple(dict.fromkeys(sets))
    if not selected or any(name not in SETS for name in selected):
        raise ValueError("Select one or more known annotated sets")
    by_set: dict[str, Counter[str]] = {}
    occurrences: dict[str, list[dict[str, Any]]] = {}
    for name, filename, document, rows in annotated_documents(data_root, run_dir, selected):
        counts, details = score_document(document, rows)
        by_set.setdefault(name, Counter()).update(counts)
        occurrences[f"{name}/{filename}"] = details
    totals: Counter[str] = Counter()
    for counts in by_set.values():
        totals.update(counts)
    return {
        "stage": STAGE,
        "basis": (
            "Only this stage's decided case-name, court, and date judgments score against explicit "
            "field labels on the same annotated root occurrence with the latest aligned reading."
        ),
        "sets": {name: _summary(counts) for name, counts in by_set.items()},
        "totals": _summary(totals),
        "occurrences": occurrences,
    }
