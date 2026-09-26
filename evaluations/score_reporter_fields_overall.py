"""Score the latest reporter-root field judgments across every completed route.

This is an overall checkpoint score, not an incremental stage score. Recall
includes every explicitly labeled canonical reporter-root field, including
roots first cited in a table of authorities, even if lookup found nothing or
no candidate was selected. Precision uses only decided judgments whose
selected record and source reading align with the root's annotation. A later
representative can carry a field label only when it states the same field.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from evaluations.annotations import SETS, annotated_documents, span
from evaluations.score_identity import (
    FIELD_LOGS,
    GOLD_FIELD_RESULT,
    _same_annotated_field,
    _same_annotated_reading,
)
from mellea_lrc.model import Document, FullReporterCitation, latest
from mellea_lrc.model.citations.judgments import MatchResult
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactAmbiguityOutcome,
    ReporterExactLookupOutcome,
)
from mellea_lrc.validation.stage_names import (
    REPORTER_ROOT_LOOKUP,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM,
    REPORTER_ROOT_LOOKUP_UNIQUE_LLM,
)

NAME = "reporter_fields_overall"
FIELDS = ("case_name", "court", "date")
FINAL_STAGES = {
    REPORTER_ROOT_LOOKUP,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM,
    REPORTER_ROOT_LOOKUP_UNIQUE_LLM,
}


def _selected_candidate(root: FullReporterCitation) -> tuple[int | None, str | None]:
    """Use the final identity node's representative, never an earlier hint."""
    if not root.identity_judgments:
        return None, None
    judgment = root.identity_judgments[-1]
    stage = next(node.stage for node in root.nodes if node.id == judgment.node_id)
    lookup = root.reporter_exact_lookup
    if lookup is None or lookup.response is None:
        return None, stage
    if stage == REPORTER_ROOT_LOOKUP:
        return (
            0 if lookup.outcome is ReporterExactLookupOutcome.UNIQUE and judgment.next_stage is None else None
        ), stage
    if stage == REPORTER_ROOT_LOOKUP_AMBIGUOUS:
        resolution = root.reporter_exact_ambiguity_resolution
        return (
            resolution.selected_candidate_index
            if resolution is not None
            and resolution.outcome is ReporterExactAmbiguityOutcome.UNIQUE_RULE_MATCH
            else None
        ), stage
    if stage == REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM:
        review = root.reporter_ambiguous_review
        return (
            review.decision.selected_candidate_index
            if review is not None and review.decision is not None
            else None
        ), stage
    if stage == REPORTER_ROOT_LOOKUP_UNIQUE_LLM:
        review = root.reporter_unique_review
        return (0 if review is not None and review.decision is not None else None), stage
    return None, stage


def _gold_cluster_ids(row: dict[str, Any]) -> set[str]:
    return {
        str(source["id"])
        for evidence in row.get("validation", {}).get("identity", {}).get("evidence", ())
        if (source := evidence.get("source", {})).get("kind") == "cluster" and source.get("id") is not None
    }


def _summary(counts: Counter[str]) -> dict[str, Any]:
    fields: dict[str, dict[str, int | float | None]] = {}
    for field in FIELDS:
        correct = counts[f"{field}_correct"]
        scored = counts[f"{field}_scored"]
        gold = counts[f"{field}_gold"]
        fields[field] = {
            "correct": correct,
            "scored": scored,
            "gold": gold,
            "precision": round(correct / scored, 4) if scored else None,
            "recall": round(correct / gold, 4) if gold else None,
        }
    return {
        "population": {
            "reporter_locator_occurrences": counts["gold_reporter_locator_occurrences"],
            "reporter_identities": counts["gold_reporter_identities"],
            "full_reporter_roots": counts["gold_canonical_reporter_roots"],
            "table_of_authorities_roots": counts["gold_table_of_authorities_roots"],
            "roots_with_lookup_clusters": counts["gold_roots_with_lookup_clusters"],
        },
        "fields": fields,
    }


def score_document(
    document: Document, rows: tuple[dict[str, Any], ...]
) -> tuple[Counter[str], list[dict[str, Any]]]:
    """Count each canonical annotated reporter field at most once."""
    if not FINAL_STAGES <= set(document.stage_runs):
        raise ValueError("Overall reporter fields require exact, ambiguous, and both model-review stages")
    if document.stage_runs[-1] not in {
        REPORTER_ROOT_LOOKUP_UNIQUE_LLM,
        REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM,
    }:
        raise ValueError("Overall reporter fields require the final model-review checkpoint")
    by_locator: dict[tuple[int, int], dict[str, Any]] = {}
    gold_roots: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("kind") != "FullCaseCitation":
            continue
        locator = row.get("locator")
        if not isinstance(locator, dict):
            continue
        key = (span(locator).start, span(locator).end)
        if key in by_locator:
            raise ValueError("Duplicate annotated reporter locator")
        by_locator[key] = row
        if row.get("is_root") is True:
            if row.get("identifier", {}).get("kind") != "reporter":
                raise ValueError("Canonical full reporter root needs a reporter identifier")
            if row.get("id") != row.get("root_id"):
                raise ValueError("Canonical root annotation has a different root_id")
            gold_roots[key] = row
    predicted_roots = {
        (root.locator_span.start, root.locator_span.end): root
        for root in document.roots
        if isinstance(root, FullReporterCitation)
    }
    if len(predicted_roots) != sum(isinstance(root, FullReporterCitation) for root in document.roots):
        raise ValueError("Duplicate predicted reporter root locator")
    member_gold_ids: dict[str, set[str]] = {}
    for member in document.full_locators:
        if not isinstance(member, FullReporterCitation):
            continue
        annotation = by_locator.get((member.locator_span.start, member.locator_span.end))
        root_id = latest(member.root_id)
        if annotation is not None and root_id is not None:
            member_gold_ids.setdefault(root_id, set()).add(annotation["root_id"])
    predicted_by_gold_id: dict[str, list[FullReporterCitation]] = {}
    for root in predicted_roots.values():
        for gold_id in member_gold_ids.get(root.id, ()):
            predicted_by_gold_id.setdefault(gold_id, []).append(root)

    counts: Counter[str] = Counter(
        documents=1,
        gold_reporter_locator_occurrences=len(by_locator),
        gold_reporter_identities=len({row["root_id"] for row in by_locator.values()}),
        gold_canonical_reporter_roots=len(gold_roots),
        gold_table_of_authorities_roots=sum(
            row.get("in_table_of_authorities") is True for row in gold_roots.values()
        ),
    )
    details: list[dict[str, Any]] = []
    for key, gold in sorted(gold_roots.items()):
        root = predicted_roots.get(key)
        linked_roots = predicted_by_gold_id.get(gold["id"], ())
        if root is None and len(linked_roots) == 1:
            root = linked_roots[0]
        representative = (
            by_locator.get((root.locator_span.start, root.locator_span.end)) if root is not None else None
        )
        selected, decision_stage = _selected_candidate(root) if root is not None else (None, None)
        candidate = (
            root.reporter_exact_lookup.response.clusters[selected]
            if root is not None
            and selected is not None
            and root.reporter_exact_lookup is not None
            and root.reporter_exact_lookup.response is not None
            else None
        )
        if root is not None and selected is not None and candidate is None:
            raise ValueError("Final reporter judgment selected an unavailable candidate")
        evidence_ids = _gold_cluster_ids(gold)
        member_ids = member_gold_ids.get(root.id, set()) if root is not None else set()
        cluster_count = (
            len(root.reporter_exact_lookup.response.clusters)
            if root is not None
            and root.reporter_exact_lookup is not None
            and root.reporter_exact_lookup.response is not None
            else 0
        )
        if (
            cluster_count
            and representative is not None
            and representative["root_id"] == gold["id"]
            and len(member_ids) == 1
        ):
            counts["gold_roots_with_lookup_clusters"] += 1
        for field in FIELDS:
            label = (
                gold.get("validation", {}).get("identity", {}).get("fields", {}).get(field, {}).get("label")
            )
            if label not in GOLD_FIELD_RESULT:
                continue
            counts[f"{field}_gold"] += 1
            record = None
            reading = None
            outcome = "ambiguous_root_mapping" if root is None and len(linked_roots) > 1 else "missing_root"
            if root is not None:
                outcome = "conflicting_gold_roots" if len(member_ids) > 1 else "no_selected_candidate"
                if representative is None or representative["root_id"] != gold["id"]:
                    outcome = "misaligned_representative"
                elif key != (root.locator_span.start, root.locator_span.end) and not _same_annotated_field(
                    field, gold, representative
                ):
                    outcome = "changed_occurrence"
                elif selected is not None and len(member_ids) <= 1:
                    identity = root.identity_judgments[-1]
                    judgments = [
                        item
                        for item in getattr(root, FIELD_LOGS[field])
                        if item.node_id == identity.node_id and item.candidate_index == selected
                    ]
                    if len(judgments) > 1:
                        raise ValueError("Final reporter node wrote more than one field judgment")
                    record = judgments[0] if judgments else None
                    readings = getattr(root, field)
                    if record is None:
                        outcome = "no_field_judgment"
                    elif record.reading_index is None or record.reading_index >= len(readings):
                        outcome = "missing_reading"
                    else:
                        reading = readings[record.reading_index]
                        if record.reading_index != len(readings) - 1:
                            outcome = "stale_reading"
                        elif not _same_annotated_reading(field, reading, representative.get(field)):
                            outcome = "misaligned_reading"
                        elif str(candidate.id) not in evidence_ids:
                            outcome = "candidate_alignment_unverified"
                        elif record.result is MatchResult.UNDETERMINED:
                            outcome = "undetermined"
                        else:
                            counts[f"{field}_scored"] += 1
                            if record.result is GOLD_FIELD_RESULT[label]:
                                counts[f"{field}_correct"] += 1
                                outcome = "correct"
                            else:
                                outcome = "incorrect"
            counts[f"{field}_outcome_{outcome}"] += 1
            details.append(
                {
                    "document_source": document.source_path,
                    "gold_root_id": gold["id"],
                    "locator_span": {"start": key[0], "end": key[1]},
                    "citation_id": root.id if root is not None else None,
                    "representative_locator_span": (
                        {
                            "start": root.locator_span.start,
                            "end": root.locator_span.end,
                        }
                        if root is not None
                        else None
                    ),
                    "field": field,
                    "gold_label": label,
                    "decision_stage": decision_stage,
                    "selected_candidate_index": selected,
                    "selected_cluster_id": candidate.id if candidate is not None else None,
                    "gold_cluster_ids": sorted(evidence_ids),
                    "lookup_cluster_count": cluster_count,
                    "reading_index": record.reading_index if record is not None else None,
                    "reading": reading.model_dump(mode="json") if reading is not None else None,
                    "result": record.result.value if record is not None else None,
                    "outcome": outcome,
                }
            )
    return counts, details


def evaluate(data_root: Path, run_dir: Path, sets: tuple[str, ...]) -> dict[str, Any]:
    """Score the combined saved run without retrieval or model calls."""
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
        "name": NAME,
        "basis": (
            "Latest selected reporter-root field judgments; precision among decided judgments "
            "with the same annotated field content, aligned reading, and selected cluster in "
            "annotation evidence. Recall over all explicitly labeled canonical "
            "reporter-root fields, including roots with no lookup or model decision."
        ),
        "sets": {name: _summary(counts) for name, counts in by_set.items()},
        "totals": _summary(totals),
        "diagnostics": {name: dict(counts) for name, counts in by_set.items()},
        "occurrences": occurrences,
    }
