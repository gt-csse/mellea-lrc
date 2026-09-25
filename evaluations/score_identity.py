"""Score reporter-root identity from a saved exact-lookup checkpoint.

Run from the repository root::

    uv run python -m evaluations.score_identity --run-dir local/reporter-exact \
        --output-dir local/evaluations/reporter-exact

This reads annotations only after prediction artifacts exist. Deferred roots
are abstentions. The recall denominator includes every labeled gold reporter
root represented by an unmasked full citation, even when extraction missed it.
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
from mellea_lrc.model.citations.judgments import IdentityJudgment, IdentityVerdict
from mellea_lrc.validation.reporter_root_exact_lookup import STAGE

LABELS = frozenset((IdentityVerdict.CORRECT_IDENTITY.name, IdentityVerdict.WRONG_IDENTITY.name))


def _locator_key(citation: FullCitation) -> tuple[str, int, int]:
    kind = "FullCaseCitation" if isinstance(citation, FullReporterCitation) else "DocketCitation"
    return kind, citation.locator_span.start, citation.locator_span.end


def _summary(counts: Counter[str]) -> dict[str, Any]:
    def ratio(numerator: str, denominator: str) -> float | None:
        total = counts[denominator]
        return round(counts[numerator] / total, 4) if total else None

    return {
        **dict(counts),
        "decision_precision": ratio("correct_decisions", "scored_decisions"),
        "reporter_root_decision_recall": ratio("correct_decisions", "gold_reporter_roots"),
        "conditional_decision_recall": ratio("correct_decisions", "reached_gold_reporter_roots"),
        "all_root_decision_recall": ratio("correct_decisions", "gold_all_roots"),
        "admission_precision": ratio("correct_admissions", "scored_admissions"),
        "admission_recall": ratio("correct_admissions", "gold_correct_reporter_roots"),
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
    )
    decisions: dict[str, list[IdentityJudgment]] = defaultdict(list)
    for item in product.records:
        if item.name == "identity_judgments":
            if not isinstance(item.citation, FullReporterCitation) or not isinstance(
                item.record, IdentityJudgment
            ):
                raise ValueError("Reporter exact lookup wrote an unexpected identity judgment")
            decisions[item.citation.id].append(item.record)

    reporter_roots = tuple(root for root in checkpoint.roots if isinstance(root, FullReporterCitation))
    if set(decisions) != {root.id for root in reporter_roots} or any(
        len(items) != 1 for items in decisions.values()
    ):
        raise ValueError("Each reporter root needs exactly one exact-lookup identity judgment")
    counts["predicted_reporter_roots"] = len(reporter_roots)
    reached: set[str] = set()
    credited: set[str] = set()
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
            counts[f"deferred_to_{judgment.next_step.value}"] += 1
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
                "citation_id": root.id,
                "locator_span": {"start": root.locator_span.start, "end": root.locator_span.end},
                "gold_root_id": gold_id,
                "gold_label": label if label in LABELS else None,
                "verdict": verdict,
                "next_step": judgment.next_step.value if judgment.next_step else None,
                "lookup_outcome": root.reporter_exact_lookup.outcome.value
                if root.reporter_exact_lookup is not None
                else None,
                "outcome": outcome,
            }
        )

    counts["reached_gold_reporter_roots"] = len(reached)
    counts["missing_gold_reporter_roots"] = len(gold_reporter) - len(reached)
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
        {key: run_spec[key] for key in ("root_rules", "hunt_dockets", "checkpoints")}
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
        "basis": "Stage-written reporter-root identity verdicts; deferred roots abstain; recall uses all labeled roots represented by unmasked full citations",
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
    result = evaluate(args.data_root, args.run_dir, tuple(args.sets or SETS))
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
