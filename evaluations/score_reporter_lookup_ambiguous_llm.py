"""Score field judgments written by ambiguous reporter model review.

Only a judgment for the model's selected candidate can score. The annotation
must label the same root occurrence, and the judged reading must be that
occurrence's latest aligned reading. Selection and identity are diagnostics,
not metrics in this stage report.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluations.annotations import SETS, annotated_documents, span
from evaluations.score_identity import FIELD_LOGS, FIELD_TYPES, GOLD_FIELD_RESULT, _same_annotated_reading
from evaluations.stage_products import stage_product
from mellea_lrc.model import Document, FullReporterCitation, latest
from mellea_lrc.model.citations.judgments import IdentityJudgment, MatchResult
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterAmbiguousReview,
    ReporterExactAmbiguityOutcome,
    ReporterExactLookupOutcome,
)
from mellea_lrc.validation.reporter_root_lookup_ambiguous import STAGE as AMBIGUOUS_STAGE
from mellea_lrc.validation.stage_names import REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM as STAGE

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
    document: Document,
    rows: tuple[dict[str, Any], ...],
    *,
    identity_roots: tuple[dict[str, Any], ...] | None = None,
) -> tuple[Counter[str], list[dict[str, Any]]]:
    """Compare only this stage's selected-candidate fields with aligned labels."""
    product = stage_product(document, STAGE)
    if product.before is None or product.before.stage_runs[-1] != AMBIGUOUS_STAGE:
        raise ValueError("Ambiguous reporter model review needs its rule-stage checkpoint")
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
        raise ValueError("A routed reporter root is missing from the ambiguous-review checkpoint")

    by_locator: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("kind") != "FullCaseCitation" or not isinstance(row.get("locator"), dict):
            continue
        locator = span(row["locator"])
        key = (locator.start, locator.end)
        if key in by_locator:
            raise ValueError("Duplicate annotated reporter locator")
        by_locator[key] = row
    canonical_roots: dict[str, dict[str, Any]] = {}
    for row in rows if identity_roots is None else identity_roots:
        if row.get("is_root") is not True or row.get("kind") != "FullCaseCitation":
            continue
        gold_id = row.get("id")
        if gold_id is None:
            continue
        if gold_id in canonical_roots:
            raise ValueError("Duplicate annotated reporter root ID")
        canonical_roots[gold_id] = row

    reviews: dict[str, list[ReporterAmbiguousReview]] = defaultdict(list)
    identities: dict[str, list[IdentityJudgment]] = defaultdict(list)
    judgments: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for item in product.records:
        if item.name not in {*FIELD_TYPES, "reporter_ambiguous_review", "identity_judgments"}:
            continue
        if not isinstance(item.citation, FullReporterCitation) or item.citation.id not in routed:
            raise ValueError("Ambiguous-review record belongs to a nonrouted root")
        if item.name == "reporter_ambiguous_review":
            if not isinstance(item.record, ReporterAmbiguousReview):
                raise ValueError("Ambiguous-review stage wrote an unexpected review record")
            reviews[item.citation.id].append(item.record)
        elif item.name == "identity_judgments":
            if not isinstance(item.record, IdentityJudgment):
                raise ValueError("Ambiguous-review stage wrote an unexpected identity judgment")
            identities[item.citation.id].append(item.record)
        else:
            if not isinstance(item.record, FIELD_TYPES[item.name]):
                raise ValueError("Ambiguous-review stage wrote an unexpected field judgment")
            judgments[item.citation.id, item.name].append(item.record)
    if set(reviews) != routed or any(len(items) != 1 for items in reviews.values()):
        raise ValueError("Each routed reporter root needs one ambiguous model review")
    if set(identities) != routed or any(len(items) != 1 for items in identities.values()):
        raise ValueError("Each routed reporter root needs one ambiguous-review identity judgment")

    counts: Counter[str] = Counter(documents=1, routed_roots=len(routed))
    details: list[dict[str, Any]] = []
    for root_id in sorted(routed):
        root = roots[root_id]
        lookup = root.reporter_exact_lookup
        resolution = root.reporter_exact_ambiguity_resolution
        if (
            lookup is None
            or lookup.outcome is not ReporterExactLookupOutcome.AMBIGUOUS
            or lookup.response is None
            or not 2 <= len(lookup.response.clusters) < 20
            or resolution is None
            or resolution.outcome is not ReporterExactAmbiguityOutcome.NO_UNIQUE_RULE_MATCH
        ):
            raise ValueError("Ambiguous-review root needs a bounded unresolved rule-stage lookup")
        review = reviews[root_id][0]
        if getattr(root, "reporter_ambiguous_review") != review:
            raise ValueError("Ambiguous review record differs from the citation checkpoint")
        decision = review.decision
        selected_index = decision.selected_candidate_index if decision is not None else None
        if selected_index is not None and not 0 <= selected_index < len(lookup.response.clusters):
            raise ValueError("Ambiguous review selected a nonexistent candidate")
        candidate = lookup.response.clusters[selected_index] if selected_index is not None else None
        selected_docket = next(
            (
                item.response
                for item in root.reporter_exact_candidate_dockets
                if item.candidate_index == selected_index
            ),
            None,
        )
        candidate_values = (
            {
                "case_name": candidate.case_name_full or candidate.case_name,
                "court": {
                    "cluster_court_id": candidate.court_id,
                    "cluster_court": candidate.court,
                    "docket_court_id": selected_docket.court_id if selected_docket is not None else None,
                    "docket_court": selected_docket.court if selected_docket is not None else None,
                },
                "date": candidate.date_filed,
            }
            if candidate is not None
            else {}
        )
        annotated = by_locator.get((root.locator_span.start, root.locator_span.end))
        gold_id = annotated.get("root_id") if annotated is not None else None
        member_gold_ids = {
            row["root_id"]
            for member in checkpoint.full_locators
            if isinstance(member, FullReporterCitation)
            if latest(member.root_id) == root.id
            if (row := by_locator.get((member.locator_span.start, member.locator_span.end))) is not None
            if row.get("root_id") is not None
        }
        canonical = canonical_roots.get(gold_id) if gold_id is not None else None
        evidence_ids = sorted(
            {
                str(source_id)
                for evidence in (canonical or {})
                .get("validation", {})
                .get("identity", {})
                .get("evidence", ())
                if isinstance(evidence, dict)
                if isinstance((source := evidence.get("source")), dict)
                if source.get("kind") == "cluster"
                if (source_id := source.get("id")) is not None and str(source_id).strip()
            }
        )
        selected_cluster_id = str(candidate.id) if candidate is not None else None
        candidate_aligned = selected_cluster_id in evidence_ids if selected_cluster_id is not None else None
        same_root_occurrence = (
            annotated is not None
            and annotated.get("is_root") is True
            and annotated.get("id") == gold_id
            and annotated.get("identifier", {}).get("kind") == "reporter"
            and len(member_gold_ids) <= 1
        )
        gold = canonical if same_root_occurrence else None
        for field in FIELDS:
            log_name = FIELD_LOGS[field]
            records = judgments.get((root_id, log_name), ())
            if len(records) > 1:
                raise ValueError(f"Ambiguous-review stage wrote multiple {field} judgments for one root")
            if selected_index is None and records:
                raise ValueError("An ambiguous review without a selection cannot write field judgments")
            if selected_index is not None and not records:
                raise ValueError(f"Selected ambiguous review omitted its {field} judgment")
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
                outcome = "review_failed" if decision is None else "no_candidate_selected"
            else:
                counts[f"{field}_judgments"] += 1
                if record.candidate_index != selected_index:
                    raise ValueError(f"Ambiguous-review {field} judgment references another candidate")
                if record.reading_index is not None and record.reading_index >= len(readings):
                    raise ValueError(f"Ambiguous-review {field} judgment has an invalid reading index")
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
                elif not candidate_aligned:
                    outcome = "candidate_alignment_unverified"
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
                    "candidate_count": len(lookup.response.clusters),
                    "selected_candidate_index": selected_index,
                    "selected_cluster_id": candidate.id if candidate is not None else None,
                    "annotated_cluster_evidence_ids": evidence_ids,
                    "selected_cluster_in_annotation_evidence": candidate_aligned,
                    "candidate_index": record.candidate_index if record is not None else None,
                    "candidate_value": candidate_values.get(field),
                    "result": record.result.value if record is not None else None,
                    "review_failure_reason": review.failure_reason,
                    "selection_exclusion_reason": (
                        decision.reason if decision is not None and selected_index is None else None
                    ),
                    "review_decision": decision.model_dump(mode="json") if decision is not None else None,
                    "identity_verdict": identities[root_id][0].verdict.value,
                    "next_stage": identities[root_id][0].next_stage,
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
        "basis": (
            "Only this stage's decided case-name, court, and date judgments for its selected candidate "
            "score against explicit field labels on the same annotated root occurrence with the "
            "latest aligned reading and a selected cluster ID listed in canonical root evidence."
        ),
        "sets": {name: _summary(counts) for name, counts in by_set.items()},
        "totals": _summary(totals),
        "occurrences": occurrences,
    }
