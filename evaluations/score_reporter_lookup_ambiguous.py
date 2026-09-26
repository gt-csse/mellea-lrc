"""Score saved decisions for reporter exact-lookup ambiguity.

This evaluator only reads completed documents and annotations. It makes no
provider or model calls. It scores selected-candidate field judgments against
root-level field labels.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluations.annotations import SETS, annotated_documents, span
from mellea_lrc.model import Document, FullReporterCitation
from mellea_lrc.model.citations.judgments import IdentityVerdict, MatchResult
from mellea_lrc.model.citations.reporter_lookup import ReporterExactAmbiguityOutcome
from mellea_lrc.validation.reporter_root_lookup_ambiguous import STAGE
from mellea_lrc.validation.stage_names import (
    REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM,
    REPORTER_ROOT_LOOKUP_LARGE_CANDIDATE_REVIEW,
)

OUTCOMES = ("unique_rule_match", "no_unique_rule_match", "candidate_limit_exceeded")
FIELD_LOGS = ("case_name_judgments", "court_judgments", "date_judgments")


def _summary(counts: Counter[str]) -> dict[str, Any]:
    def ratio(numerator: str, denominator: str) -> float | None:
        total = counts[denominator]
        return round(counts[numerator] / total, 4) if total else None

    return {
        "field_precision": {
            field: {
                "value": ratio(f"{field}_correct", f"{field}_scored"),
                "correct": counts[f"{field}_correct"],
                "scored": counts[f"{field}_scored"],
            }
            for field in ("case_name", "court", "date")
        },
    }


def score_document(
    document: Document,
    rows: tuple[dict[str, Any], ...],
    *,
    identity_roots: tuple[dict[str, Any], ...] | None = None,
) -> tuple[Counter[str], list[dict[str, Any]]]:
    """Score one ambiguity-stage checkpoint and return occurrence evidence."""
    checkpoint = document.get_stage(STAGE)
    previous = document.get_stage(checkpoint.stage_runs[-2])
    routed_ids = {
        root.id
        for root in previous.roots
        if isinstance(root, FullReporterCitation)
        and root.identity_judgments
        and root.identity_judgments[-1].next_stage == STAGE
    }
    reporter_rows: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        # A predicted root may correspond to an annotated repeated occurrence;
        # only the annotation's canonical root carries the identifier/identity.
        if row.get("kind") == "FullCaseCitation" and isinstance(row.get("locator"), dict):
            key = (span(row["locator"]).start, span(row["locator"]).end)
            if key in reporter_rows:
                raise ValueError("Duplicate annotated reporter locator")
            reporter_rows[key] = row

    root_rows = rows if identity_roots is None else identity_roots
    gold_by_id = {
        row["id"]: row for row in root_rows if row.get("is_root") and row.get("kind") == "FullCaseCitation"
    }
    if len(gold_by_id) != sum(
        bool(row.get("is_root") and row.get("kind") == "FullCaseCitation") for row in root_rows
    ):
        raise ValueError("Duplicate annotated reporter root ID")
    counts: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    for root in checkpoint.roots:
        if root.id not in routed_ids:
            continue
        if not isinstance(root, FullReporterCitation):
            raise ValueError(f"Routed reporter citation changed type: {root.id}")
        lookup = root.reporter_exact_lookup
        if lookup is None or lookup.outcome.value != "ambiguous":
            raise ValueError(f"Routed reporter citation lacks an ambiguous lookup: {root.id}")
        annotated = reporter_rows.get((root.locator_span.start, root.locator_span.end))
        gold_id = annotated.get("root_id") if annotated else None
        gold = gold_by_id.get(gold_id) if gold_id else None
        label = gold.get("validation", {}).get("identity", {}).get("label") if gold else None

        resolution = root.reporter_exact_ambiguity_resolution
        if resolution is None:
            raise ValueError(f"Ambiguous reporter root lacks a stage resolution: {root.id}")

        outcome = resolution.outcome.value
        if outcome not in OUTCOMES:
            raise ValueError(f"Unknown ambiguity outcome: {outcome}")
        if not root.identity_judgments or root.identity_judgments[-1].node_id != resolution.node_id:
            raise ValueError("Ambiguity resolution needs its own identity judgment")
        judgment = root.identity_judgments[-1]
        if outcome == ReporterExactAmbiguityOutcome.UNIQUE_RULE_MATCH.value:
            if judgment.verdict is not IdentityVerdict.CORRECT_IDENTITY:
                raise ValueError("A unique rule match must admit the root identity")
        elif judgment.verdict is not IdentityVerdict.DEFERRED or judgment.next_stage != (
            REPORTER_ROOT_LOOKUP_LARGE_CANDIDATE_REVIEW
            if outcome == "candidate_limit_exceeded"
            else REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM
        ):
            raise ValueError("Deferred ambiguity outcome has the wrong route")
        candidates = []
        candidate_total = len(lookup.response.clusters) if lookup.response else 0
        annotated_cluster_ids = {
            str(evidence.get("source", {}).get("id"))
            for evidence in (gold or {}).get("validation", {}).get("identity", {}).get("evidence", ())
            if evidence.get("source", {}).get("kind") == "cluster"
            and evidence.get("source", {}).get("id") is not None
        }
        selected_cluster_id = (
            lookup.response.clusters[resolution.selected_candidate_index].id
            if resolution.selected_candidate_index is not None and lookup.response is not None
            else None
        )
        selected_index = resolution.selected_candidate_index
        if selected_index is not None and gold is not None:
            fields = gold.get("validation", {}).get("identity", {}).get("fields", {})
            for field in ("case_name", "court", "date"):
                gold_field = fields.get(field, {})
                gold_label = gold_field.get("label")
                if gold_label not in {"agrees", "disagrees"}:
                    continue
                selected_judgments = [
                    item
                    for item in getattr(root, f"{field}_judgments")
                    if item.candidate_index == selected_index
                    and any(node.id == item.node_id and node.stage == STAGE for node in root.nodes)
                ]
                if len(selected_judgments) > 1:
                    raise ValueError(f"Multiple selected-candidate {field} judgments at ambiguity stage")
                if not selected_judgments:
                    continue
                prediction = selected_judgments[0].result
                if prediction is MatchResult.UNDETERMINED:
                    continue
                counts[f"{field}_scored"] += 1
                if prediction is (MatchResult.MATCH if gold_label == "agrees" else MatchResult.MISMATCH):
                    counts[f"{field}_correct"] += 1
        for index in range(candidate_total):
            field_judgments = {}
            for log_name in FIELD_LOGS:
                field_judgments[log_name.removesuffix("_judgments")] = [
                    {
                        "reading_index": item.reading_index,
                        "candidate_index": item.candidate_index,
                        "result": item.result.value,
                    }
                    for item in getattr(root, log_name)
                    if item.candidate_index == index
                    and any(node.id == item.node_id and node.stage == STAGE for node in root.nodes)
                ]
            candidates.append(
                {
                    "candidate_index": index,
                    "cluster_id": lookup.response.clusters[index].id,
                    "field_judgments": field_judgments,
                }
            )
        details.append(
            {
                "citation_id": root.id,
                "gold_root_id": gold_id,
                "gold_label": label,
                "candidate_count": candidate_total,
                "outcome": outcome,
                "passing_candidate_indices": list(resolution.passing_candidate_indices),
                "selected_candidate_index": resolution.selected_candidate_index,
                "selected_cluster_id": selected_cluster_id,
                "selected_cluster_in_annotation_evidence": (
                    selected_cluster_id in annotated_cluster_ids
                    if selected_cluster_id is not None and annotated_cluster_ids
                    else None
                ),
                "identity_verdict": judgment.verdict.value,
                "next_stage": judgment.next_stage,
                "candidates": candidates,
            }
        )

    if len(details) != len(routed_ids):
        raise ValueError("An ambiguity-stage route has no resulting citation")
    return counts, details


def evaluate(data_root: Path, run_dir: Path, sets: tuple[str, ...]) -> dict[str, Any]:
    """Score saved ambiguity-stage documents without external calls."""
    selected = tuple(dict.fromkeys(sets))
    if not selected or any(name not in SETS for name in selected):
        raise ValueError("Select one or more known annotated sets")
    by_set: dict[str, Counter[str]] = {}
    occurrences: dict[str, list[dict[str, Any]]] = {}
    for name, filename, document, rows in annotated_documents(data_root, run_dir, selected):
        annotation_path = data_root / name / "documents" / f"{Path(filename).stem}.jsonl"
        identity_roots = tuple(
            row
            for line in annotation_path.read_text(encoding="utf-8").splitlines()[1:]
            if (row := json.loads(line)).get("unit") == "citation" and row.get("is_root")
        )
        counts, details = score_document(document, rows, identity_roots=identity_roots)
        by_set.setdefault(name, Counter()).update(counts)
        occurrences[f"{name}/{filename}"] = details
    totals: Counter[str] = Counter()
    for counts in by_set.values():
        totals.update(counts)
    return {
        "stage": STAGE,
        "basis": "Field precision compares the selected candidate's stored case-name, court, and date judgments with explicit root-level field labels, without scoring extraction spans.",
        "sets": {name: _summary(counts) for name, counts in by_set.items()},
        "totals": _summary(totals),
        "occurrences": occurrences,
    }
