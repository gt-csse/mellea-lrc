"""Score reporter-root identity from a saved exact-lookup checkpoint.

Run from the repository root::

    uv run python -m evaluations.score_identity --run-dir local/reporter-exact \
        --output-dir local/evaluations/reporter-exact

This reads annotations only after prediction artifacts exist. Deferred roots
are abstentions. Identity recall includes every labeled gold reporter root
represented by an unmasked full citation, even when extraction missed it.
Field comparison accuracy needs the annotated root occurrence and an aligned
field reading; later representative occurrences are not assigned that label.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluations.annotations import SETS, annotated_documents, span
from evaluations.stage_products import stage_product
from mellea_lrc.model import Document, FullDocketCitation, FullReporterCitation, latest
from mellea_lrc.model.citations import FullCitation
from mellea_lrc.model.citations.fields.base import CitationField
from mellea_lrc.model.citations.judgments import (
    IdentityJudgment,
    IdentityVerdict,
    MatchResult,
    ReporterExactCaseNameJudgment,
    ReporterExactCourtJudgment,
    ReporterExactDateJudgment,
)
from mellea_lrc.model.citations.reporter_lookup import ReporterExactLookupOutcome
from mellea_lrc.validation._support.reporter_exact_fields import locator_present
from mellea_lrc.validation.reporter_root_lookup import STAGE

LABELS = frozenset((IdentityVerdict.CORRECT_IDENTITY.name, IdentityVerdict.WRONG_IDENTITY.name))
FIELD_LOGS = {
    "case_name": "case_name_judgments",
    "court": "court_judgments",
    "date": "date_judgments",
}
FIELD_TYPES = {
    "case_name_judgments": ReporterExactCaseNameJudgment,
    "court_judgments": ReporterExactCourtJudgment,
    "date_judgments": ReporterExactDateJudgment,
}
GOLD_FIELD_RESULT = {"agrees": MatchResult.MATCH, "disagrees": MatchResult.MISMATCH}


def _locator_key(citation: FullCitation) -> tuple[str, int, int]:
    kind = "FullCaseCitation" if isinstance(citation, FullReporterCitation) else "DocketCitation"
    return kind, citation.locator_span.start, citation.locator_span.end


def _same_annotated_reading(field: str, reading: CitationField[Any], gold: Any) -> bool:
    """Keep reading and normalization errors out of comparison accuracy."""
    if not isinstance(gold, dict):
        return False
    if "start" in gold and "end" in gold:
        if reading.span != span(gold) or reading.quote != gold.get("quote"):
            return False
    elif field != "court" or reading.span is not None:
        return False
    if field == "court":
        return reading.normalizable and reading.get_normalized().id == gold.get("id")
    if field == "date":
        if not reading.normalizable:
            return False
        normalized = reading.get_normalized()
        value = f"{normalized.year:04d}"
        if normalized.month is not None:
            value += f"-{normalized.month:02d}"
        if normalized.day is not None:
            value += f"-{normalized.day:02d}"
        return value == gold.get("normalized")
    return True


def _summary(counts: Counter[str]) -> dict[str, Any]:
    def ratio(numerator: str, denominator: str) -> float | None:
        total = counts[denominator]
        return round(counts[numerator] / total, 4) if total else None

    fields = {}
    for field in FIELD_LOGS:

        def count(name: str) -> int:
            return counts[f"{field}_{name}"]

        gold_stated = count("gold_agrees") + count("gold_disagrees")
        fields[field] = {
            "gold_stated": gold_stated,
            "gold_agrees": count("gold_agrees"),
            "gold_disagrees": count("gold_disagrees"),
            "gold_not_stated": count("gold_not_stated"),
            "gold_unlabeled": count("gold_unlabeled"),
            "unique_gold": count("unique_gold"),
            "eligible_gold": count("eligible_gold"),
            "missing_reading": count("missing_reading"),
            "misaligned_reading": count("misaligned_reading"),
            "judgments": count("judgments"),
            "decided": count("decided"),
            "correct_predictions": count("correct_predictions"),
            "incorrect_predictions": count("incorrect_predictions"),
            "undetermined": count("undetermined"),
            "omitted_gold": count("omitted_gold"),
            "not_stated_judgments": count("not_stated_judgments"),
            "unscored_judgments": count("unscored_judgments"),
            "changed_occurrence_judgments": count("changed_occurrence_judgments"),
            "misaligned_judgments": count("misaligned_judgments"),
            "unmatched_or_unlabeled_judgments": count("unmatched_or_unlabeled_judgments"),
            "correct_gold": count("correct_gold"),
            "precision": ratio(f"{field}_correct_predictions", f"{field}_decided"),
            "conditional_recall": ratio(f"{field}_correct_gold", f"{field}_eligible_gold"),
            "unique_lookup_recall": ratio(f"{field}_correct_gold", f"{field}_unique_gold"),
            "global_recall": round(count("correct_gold") / gold_stated, 4) if gold_stated else None,
            "confusion": {
                gold: {result.value: count(f"gold_{gold}_pred_{result.value}") for result in MatchResult}
                for gold in GOLD_FIELD_RESULT
            },
        }

    return {
        **{
            key: value
            for key, value in counts.items()
            if not key.startswith(tuple(f"{field}_" for field in FIELD_LOGS))
        },
        "decision_precision": ratio("correct_decisions", "scored_decisions"),
        "reporter_root_decision_recall": ratio("correct_decisions", "gold_reporter_roots"),
        "conditional_decision_recall": ratio("correct_decisions", "reached_gold_reporter_roots"),
        "all_root_decision_recall": ratio("correct_decisions", "gold_all_roots"),
        "admission_precision": ratio("correct_admissions", "scored_admissions"),
        "admission_recall": ratio("correct_admissions", "gold_correct_reporter_roots"),
        "fields": fields,
    }


def score_document(
    document: Document,
    gold_rows: tuple[dict[str, Any], ...],
    *,
    identity_roots: tuple[dict[str, Any], ...] | None = None,
    identity_labeled: bool = True,
) -> tuple[Counter[str], list[dict[str, Any]]]:
    """Compare exact-lookup judgments with active gold roots.

    ``gold_rows`` contains only unmasked occurrences. ``identity_roots`` may
    also contain masked root rows so an unmasked later citation can inherit its
    annotated identity without treating the masked locator as a prediction.
    """
    product = stage_product(document, STAGE)
    checkpoint = product.after
    by_locator: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in gold_rows:
        if row.get("kind") not in {"FullCaseCitation", "DocketCitation"}:
            continue
        key = (row["kind"], span(row["locator"]).start, span(row["locator"]).end)
        if key in by_locator:
            raise ValueError("Duplicate annotated full citation")
        by_locator[key] = row
    root_rows = gold_rows if identity_roots is None else identity_roots
    by_id = {row["id"]: row for row in root_rows if row.get("is_root")}
    if len(by_id) != sum(bool(row.get("is_root")) for row in root_rows):
        raise ValueError("Duplicate annotated root ID")
    active_ids = {row["root_id"] for row in by_locator.values()}
    missing_ids = active_ids - by_id.keys()
    if missing_ids:
        raise ValueError(f"Unmasked full citations point to missing annotated roots: {sorted(missing_ids)}")
    active_roots = {root_id: by_id[root_id] for root_id in active_ids}

    gold_reporter = {
        root_id: row
        for root_id, row in active_roots.items()
        if row.get("kind") == "FullCaseCitation"
        and row.get("identifier", {}).get("kind") == "reporter"
        and row.get("validation", {}).get("identity", {}).get("label") in LABELS
    }
    gold_all = {
        root_id: row
        for root_id, row in active_roots.items()
        if row.get("kind") in {"FullCaseCitation", "DocketCitation"}
        and row.get("validation", {}).get("identity", {}).get("label") in LABELS
    }
    # A root's field labels describe that occurrence, not every later citation
    # attached to it. A masked root may still supply identity gold through an
    # unmasked leaf, but its field labels cannot score that leaf's readings.
    field_gold = {
        root_id: row
        for root_id, row in gold_reporter.items()
        if (occurrence := by_locator.get((row["kind"], span(row["locator"]).start, span(row["locator"]).end)))
        and occurrence["id"] == root_id
    }
    counts: Counter[str] = Counter(
        documents=1,
        identity_labeled_documents=int(identity_labeled),
        gold_reporter_roots=len(gold_reporter),
        gold_correct_reporter_roots=sum(
            row["validation"]["identity"]["label"] == IdentityVerdict.CORRECT_IDENTITY.name
            for row in gold_reporter.values()
        ),
        gold_wrong_reporter_roots=sum(
            row["validation"]["identity"]["label"] == IdentityVerdict.WRONG_IDENTITY.name
            for row in gold_reporter.values()
        ),
        gold_all_roots=len(gold_all),
        field_gold_roots=len(field_gold),
    )
    for row in field_gold.values():
        for field in FIELD_LOGS:
            label = (
                row.get("validation", {}).get("identity", {}).get("fields", {}).get(field, {}).get("label")
            )
            if label not in {*GOLD_FIELD_RESULT, "not_stated", None}:
                raise ValueError(f"Unknown annotated {field} judgment: {label}")
            counts[f"{field}_gold_{label or 'unlabeled'}"] += 1
    decisions: dict[str, list[IdentityJudgment]] = defaultdict(list)
    field_records: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for item in product.records:
        if item.name == "identity_judgments":
            if not isinstance(item.citation, FullReporterCitation) or not isinstance(
                item.record, IdentityJudgment
            ):
                raise ValueError("Reporter exact lookup wrote an unexpected identity judgment")
            decisions[item.citation.id].append(item.record)
        elif item.name in FIELD_TYPES:
            if not isinstance(item.citation, FullReporterCitation) or not isinstance(
                item.record, FIELD_TYPES[item.name]
            ):
                raise ValueError("Reporter exact lookup wrote an unexpected field judgment")
            field_records[item.citation.id, item.name].append(item.record)

    reporter_roots = tuple(root for root in checkpoint.roots if isinstance(root, FullReporterCitation))
    if set(decisions) != {root.id for root in reporter_roots} or any(
        len(items) != 1 for items in decisions.values()
    ):
        raise ValueError("Each reporter root needs exactly one exact-lookup identity judgment")
    counts["predicted_reporter_roots"] = len(reporter_roots)
    reached: set[str] = set()
    credited: set[str] = set()
    unique_fields = {field: set() for field in FIELD_LOGS}
    aligned_fields = {field: set() for field in FIELD_LOGS}
    judged_fields = {field: set() for field in FIELD_LOGS}
    correct_fields = {field: set() for field in FIELD_LOGS}
    details: list[dict[str, Any]] = []
    for root in reporter_roots:
        members = (citation for citation in checkpoint.full_locators if latest(citation.root_id) == root.id)
        gold_ids = {
            by_locator[_locator_key(member)]["root_id"]
            for member in members
            if _locator_key(member) in by_locator
        }
        representative = by_locator.get(_locator_key(root))
        gold_id = representative["root_id"] if representative is not None else None
        gold = by_id.get(gold_id) if gold_id else None
        label = gold.get("validation", {}).get("identity", {}).get("label") if gold else None
        judgment = decisions[root.id][0]
        verdict = judgment.verdict.name
        decided = judgment.verdict is not IdentityVerdict.DEFERRED

        if len(gold_ids) > 1:
            outcome = "conflicting_gold_roots"
        elif representative is None:
            outcome = "unmatched_locator"
        elif gold is None:
            raise ValueError(f"Annotated root unexpectedly missing: {gold_id}")
        elif gold_id not in gold_reporter and label in LABELS:
            outcome = "other_labeled_root"
        elif gold_id not in gold_reporter:
            outcome = "other_or_unlabeled_root"
        elif not decided:
            outcome = "deferred"
        elif gold_id in credited:
            outcome = "duplicate_decision"
        elif verdict == label:
            outcome = "correct"
            credited.add(gold_id)
        else:
            outcome = "wrong"

        if gold_id in gold_reporter and outcome != "conflicting_gold_roots":
            reached.add(gold_id)
        counts[outcome] += 1
        if judgment.verdict is IdentityVerdict.DEFERRED:
            counts["deferred_total"] += 1
            counts[f"deferred_to_{judgment.next_stage}"] += 1
        else:
            counts["decided_total"] += 1
            if outcome in {"correct", "wrong", "duplicate_decision", "other_labeled_root"} or (
                outcome in {"unmatched_locator", "conflicting_gold_roots"} and identity_labeled
            ):
                counts["scored_decisions"] += 1
                if outcome == "correct":
                    counts["correct_decisions"] += 1
                else:
                    counts["incorrect_decisions"] += 1
                if judgment.verdict is IdentityVerdict.CORRECT_IDENTITY:
                    counts["scored_admissions"] += 1
                    if outcome == "correct":
                        counts["correct_admissions"] += 1
            else:
                counts["unscored_decisions"] += 1
        details.append(
            {
                "product": "identity_judgment",
                "citation_id": root.id,
                "locator_span": {"start": root.locator_span.start, "end": root.locator_span.end},
                "gold_root_id": gold_id,
                "gold_label": label if label in LABELS else None,
                "verdict": verdict,
                "next_stage": judgment.next_stage,
                "lookup_outcome": root.reporter_exact_lookup.outcome.value
                if root.reporter_exact_lookup is not None
                else None,
                "outcome": outcome,
            }
        )

        lookup = root.reporter_exact_lookup
        unique_lookup = lookup is not None and lookup.outcome is ReporterExactLookupOutcome.UNIQUE
        candidate = None
        if unique_lookup:
            if lookup.query is None or lookup.response is None or len(lookup.response.clusters) != 1:
                raise ValueError("Unique reporter lookup needs one saved query and cluster")
            candidate = lookup.response.clusters[0]
            membership = locator_present(candidate, lookup.query)
            counts[
                "locator_membership_match"
                if membership is True
                else "locator_membership_mismatch"
                if membership is False
                else "locator_membership_unknown"
            ] += 1
            details.append(
                {
                    "product": "locator_membership",
                    "citation_id": root.id,
                    "gold_root_id": gold_id,
                    "gold_identity_label": label if label in LABELS else None,
                    "result": "match"
                    if membership is True
                    else "mismatch"
                    if membership is False
                    else "unknown",
                    "query": lookup.query.model_dump(mode="json"),
                    "candidate_cluster_id": candidate.id,
                    "candidate_citations": [
                        citation.model_dump(mode="json") for citation in candidate.citations
                    ],
                }
            )
        comparable_id = (
            gold_id
            if outcome != "conflicting_gold_roots"
            and gold_id in field_gold
            and representative is not None
            and representative["id"] == gold_id
            else None
        )
        if comparable_id is None:
            counts["predicted_roots_without_field_gold"] += 1
        for field, log_name in FIELD_LOGS.items():
            gold_record = field_gold[comparable_id].get(field) if comparable_id is not None else None
            gold_field = (
                field_gold[comparable_id]
                .get("validation", {})
                .get("identity", {})
                .get("fields", {})
                .get(field, {})
                .get("label")
                if comparable_id is not None
                else None
            )
            readings = getattr(root, field)
            aligned_reading = bool(readings) and _same_annotated_reading(field, readings[-1], gold_record)
            if unique_lookup and gold_field in GOLD_FIELD_RESULT:
                unique_fields[field].add(comparable_id)
                if not readings:
                    counts[f"{field}_missing_reading"] += 1
                elif aligned_reading:
                    aligned_fields[field].add(comparable_id)
                else:
                    counts[f"{field}_misaligned_reading"] += 1
            judgments = field_records.get((root.id, log_name), ())
            if len(judgments) > 1:
                raise ValueError(f"Exact lookup wrote multiple {field} judgments for one root")
            if not judgments:
                if unique_lookup and gold_field in GOLD_FIELD_RESULT:
                    details.append(
                        {
                            "product": "field_gap",
                            "citation_id": root.id,
                            "gold_root_id": comparable_id,
                            "field": field,
                            "gold_label": gold_field,
                            "reason": "missing_reading"
                            if not readings
                            else "misaligned_reading"
                            if not aligned_reading
                            else "omitted_judgment",
                        }
                    )
                continue
            if not unique_lookup:
                raise ValueError(f"Exact lookup judged {field} without a unique candidate")
            judgment = judgments[0]
            if judgment.reading_index >= len(readings) or judgment.candidate_index != 0:
                raise ValueError(f"Exact lookup {field} judgment has an invalid reference")
            counts[f"{field}_judgments"] += 1
            candidate_value = {
                "case_name": candidate.case_name_full,
                "court": {
                    "cluster_court_id": candidate.court_id,
                    "cluster_court": candidate.court,
                    "docket_id": root.reporter_exact_docket.docket_id
                    if root.reporter_exact_docket is not None
                    else None,
                    "docket_court_id": root.reporter_exact_docket.response.court_id
                    if root.reporter_exact_docket is not None
                    and root.reporter_exact_docket.response is not None
                    else None,
                    "docket_court": root.reporter_exact_docket.response.court
                    if root.reporter_exact_docket is not None
                    and root.reporter_exact_docket.response is not None
                    else None,
                },
                "date": candidate.date_filed,
            }[field]
            if comparable_id is None:
                field_outcome = (
                    "changed_occurrence" if gold_id in gold_reporter else "unmatched_or_unlabeled_root"
                )
                counts[f"{field}_unscored_judgments"] += 1
                counts[
                    f"{field}_changed_occurrence_judgments"
                    if gold_id in gold_reporter
                    else f"{field}_unmatched_or_unlabeled_judgments"
                ] += 1
            elif gold_field == "not_stated":
                field_outcome = "not_stated"
                counts[f"{field}_not_stated_judgments"] += 1
            elif gold_field not in GOLD_FIELD_RESULT:
                field_outcome = "unlabeled"
                counts[f"{field}_unscored_judgments"] += 1
                counts[f"{field}_unmatched_or_unlabeled_judgments"] += 1
            elif not aligned_reading or judgment.reading_index != len(readings) - 1:
                field_outcome = "misaligned_reading"
                counts[f"{field}_unscored_judgments"] += 1
                counts[f"{field}_misaligned_judgments"] += 1
            else:
                judged_fields[field].add(comparable_id)
                counts[f"{field}_gold_{gold_field}_pred_{judgment.result.value}"] += 1
                if judgment.result is MatchResult.UNDETERMINED:
                    field_outcome = "undetermined"
                    counts[f"{field}_undetermined"] += 1
                else:
                    counts[f"{field}_decided"] += 1
                    if judgment.result is GOLD_FIELD_RESULT[gold_field]:
                        field_outcome = "correct"
                        counts[f"{field}_correct_predictions"] += 1
                        correct_fields[field].add(comparable_id)
                    else:
                        field_outcome = "incorrect"
                        counts[f"{field}_incorrect_predictions"] += 1
            details.append(
                {
                    "product": "field_judgment",
                    "citation_id": root.id,
                    "field": field,
                    "reading_index": judgment.reading_index,
                    "candidate_index": judgment.candidate_index,
                    "result": judgment.result.value,
                    "gold_root_id": comparable_id,
                    "gold_label": gold_field,
                    "gold_reading": gold_record,
                    "source_reading": readings[judgment.reading_index].model_dump(mode="json"),
                    "candidate_value": candidate_value,
                    "candidate_cluster_id": candidate.id,
                    "outcome": field_outcome,
                }
            )

    counts["reached_gold_reporter_roots"] = len(reached)
    counts["missing_gold_reporter_roots"] = len(gold_reporter) - len(reached)
    for field in FIELD_LOGS:
        counts[f"{field}_unique_gold"] = len(unique_fields[field])
        counts[f"{field}_eligible_gold"] = len(aligned_fields[field])
        counts[f"{field}_correct_gold"] = len(correct_fields[field])
        counts[f"{field}_omitted_gold"] = len(aligned_fields[field] - judged_fields[field])
        details.extend(
            {
                "product": "field_gold_miss",
                "field": field,
                "gold_root_id": gold_id,
                "gold_label": row["validation"]["identity"]["fields"][field]["label"],
                "reason": "no_unique_comparable_lookup"
                if gold_id not in unique_fields[field]
                else "missing_or_misaligned_reading"
                if gold_id not in aligned_fields[field]
                else "omitted_judgment"
                if gold_id not in judged_fields[field]
                else "undetermined_or_incorrect_judgment",
            }
            for gold_id, row in field_gold.items()
            if row.get("validation", {}).get("identity", {}).get("fields", {}).get(field, {}).get("label")
            in GOLD_FIELD_RESULT
            and gold_id not in correct_fields[field]
        )
    details.extend(
        {
            "product": "gold_miss",
            "gold_root_id": gold_id,
            "gold_label": row["validation"]["identity"]["label"],
            "locator_span": {"start": row["locator"]["start"], "end": row["locator"]["end"]},
            "reason": "missing_root" if gold_id not in reached else "not_correctly_decided",
        }
        for gold_id, row in gold_reporter.items()
        if gold_id not in credited
    )
    return counts, details


def evaluate(data_root: Path, run_dir: Path, sets: tuple[str, ...]) -> dict[str, Any]:
    """Score saved documents without making provider or model calls."""
    run_spec_path = run_dir / "run.json"
    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8")) if run_spec_path.exists() else None
    run_configuration = (
        {
            **{key: run_spec[key] for key in ("root_rules", "hunt_dockets", "checkpoints")},
            "court_docket_fetch": bool(run_spec.get("court_docket_fetch", False)),
        }
        if run_spec is not None
        else None
    )
    by_set: dict[str, Counter[str]] = {}
    occurrences: dict[str, list[dict[str, Any]]] = {}
    inputs = tuple(annotated_documents(data_root, run_dir, sets))
    roots_by_document: dict[tuple[str, str], tuple[dict[str, Any], ...]] = {}
    for name, filename, _, _ in inputs:
        annotation_path = data_root / name / "documents" / f"{Path(filename).stem}.jsonl"
        roots_by_document[name, filename] = tuple(
            row
            for line in annotation_path.read_text(encoding="utf-8").splitlines()[1:]
            if (row := json.loads(line)).get("unit") == "citation" and row.get("is_root")
        )
    identity_labeled_sets = {
        name
        for (name, _), root_rows in roots_by_document.items()
        if any(row.get("validation", {}).get("identity", {}).get("label") in LABELS for row in root_rows)
    }
    for name, filename, document, rows in inputs:
        counts, details = score_document(
            document,
            rows,
            identity_roots=roots_by_document[name, filename],
            identity_labeled=name in identity_labeled_sets,
        )
        by_set.setdefault(name, Counter()).update(counts)
        occurrences[f"{name}/{filename}"] = details
    totals: Counter[str] = Counter()
    for counts in by_set.values():
        totals.update(counts)
    return {
        "stage": STAGE,
        "prediction_run": run_configuration,
        "basis": (
            "Stage-written reporter-root identity and field judgments. Deferred identity roots abstain; "
            "identity recall uses all labeled roots represented by unmasked full citations. Field judgments "
            "are scored only against labels for the same annotated root occurrence and aligned field reading."
        ),
        "sets": {name: _summary(counts) for name, counts in by_set.items()},
        "totals": _summary(totals),
        "occurrences": occurrences,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--set", dest="sets", action="append", choices=SETS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = evaluate(args.data_root, args.run_dir, tuple(args.sets or ("primary",)))
    occurrences = result.pop("occurrences")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "occurrences.json").write_text(
            json.dumps(occurrences, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
