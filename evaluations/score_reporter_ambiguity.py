"""Score saved decisions for reporter exact-lookup ambiguity.

This evaluator only reads completed documents and annotations. It makes no
provider or model calls and scores admission against the root-level identity
annotation, never against candidate-specific field labels.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluations.annotations import SETS, annotated_documents, span
from mellea_lrc.model import Document, FullReporterCitation
from mellea_lrc.model.citations.judgments import IdentityNextStep, IdentityVerdict
from mellea_lrc.model.citations.reporter_lookup import ReporterExactAmbiguityOutcome
from mellea_lrc.validation.reporter_root_exact_ambiguity import STAGE

OUTCOMES = ("unique_rule_match", "review_required", "too_many_candidates")
GOLD_LABELS = {"CORRECT_IDENTITY", "WRONG_IDENTITY"}
FIELD_LOGS = ("case_name_judgments", "court_judgments", "date_judgments")


def _summary(counts: Counter[str]) -> dict[str, Any]:
    def ratio(numerator: str, denominator: str) -> float | None:
        total = counts[denominator]
        return round(counts[numerator] / total, 4) if total else None

    return {
        **dict(counts),
        "admission_precision": ratio("correct_admissions", "predicted_admissions"),
        "admission_recall": ratio("correct_admissions", "gold_correct_roots"),
        "route_outcomes": {outcome: counts[f"outcome_{outcome}"] for outcome in OUTCOMES},
    }


def score_document(
    document: Document,
    rows: tuple[dict[str, Any], ...],
    *,
    identity_roots: tuple[dict[str, Any], ...] | None = None,
) -> tuple[Counter[str], list[dict[str, Any]]]:
    """Score one ambiguity-stage checkpoint and return occurrence evidence."""
    checkpoint = document.get_stage(STAGE)
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
        row["id"]: row
        for row in root_rows
        if row.get("is_root") and row.get("kind") == "FullCaseCitation"
    }
    if len(gold_by_id) != sum(
        bool(row.get("is_root") and row.get("kind") == "FullCaseCitation") for row in root_rows
    ):
        raise ValueError("Duplicate annotated reporter root ID")
    counts: Counter[str] = Counter(documents=1)
    details: list[dict[str, Any]] = []
    seen_gold: set[str] = set()
    correct_gold: set[str] = set()
    credited_gold: set[str] = set()
    for root in checkpoint.roots:
        if not isinstance(root, FullReporterCitation):
            continue
        lookup = root.reporter_exact_lookup
        if lookup is None or lookup.outcome.value != "ambiguous":
            continue
        annotated = reporter_rows.get((root.locator_span.start, root.locator_span.end))
        gold_id = annotated.get("root_id") if annotated else None
        gold = gold_by_id.get(gold_id) if gold_id else None
        label = gold.get("validation", {}).get("identity", {}).get("label") if gold else None
        if label in GOLD_LABELS and gold_id not in seen_gold:
            seen_gold.add(gold_id)
            if label == "CORRECT_IDENTITY":
                correct_gold.add(gold_id)

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
        elif judgment.verdict is not IdentityVerdict.DEFERRED or judgment.next_step is not (
            IdentityNextStep.FUTURE_IMPLEMENTATION
            if outcome == ReporterExactAmbiguityOutcome.TOO_MANY_CANDIDATES.value
            else IdentityNextStep.AMBIGUITY
        ):
            raise ValueError("Deferred ambiguity outcome has the wrong route")
        counts[f"outcome_{outcome}"] += 1
        if outcome == "unique_rule_match":
            counts["predicted_admissions"] += 1
            if label == "CORRECT_IDENTITY" and gold_id not in credited_gold:
                counts["correct_admissions"] += 1
                credited_gold.add(gold_id)

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
        if selected_cluster_id is not None:
            if annotated_cluster_ids:
                counts["selection_with_cluster_evidence"] += 1
                if selected_cluster_id in annotated_cluster_ids:
                    counts["selected_cluster_listed_in_evidence"] += 1
            else:
                counts["selection_without_cluster_evidence"] += 1
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
            candidates.append({
                "candidate_index": index,
                "cluster_id": lookup.response.clusters[index].id,
                "field_judgments": field_judgments,
            })
        details.append({
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
            "candidates": candidates,
        })

    counts["gold_ambiguous_roots"] = len(seen_gold)
    counts["gold_correct_roots"] = len(correct_gold)
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
        "basis": "Ambiguous reporter roots admitted by unique_rule_match, scored against root-level identity labels; candidate field labels are not used.",
        "sets": {name: _summary(counts) for name, counts in by_set.items()},
        "totals": _summary(totals),
        "occurrences": occurrences,
    }
