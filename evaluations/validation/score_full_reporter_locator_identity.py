"""Score a persisted full-reporter-locator identity checkpoint.

This evaluator reads only completed ``Document`` artifacts and annotation
rows.  It does not replay extraction, field parsing, lookup, or model calls.
The reported admission score treats ``resolved`` as the sole positive identity
claim; every deferred outcome remains visible in the outcome-by-gold matrix.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluations.extraction.locator_eval_common import score_sets

_IDENTITY_LABELS = frozenset({"CORRECT_IDENTITY", "WRONG_IDENTITY"})


def score_full_reporter_locator_identity(
    *,
    annotations: Path,
    artifacts: Path,
) -> dict[str, object]:
    """Score one root-identity artifact directory against annotated roots."""
    artifact_documents = {path.stem for path in artifacts.glob("*.json")}
    gold = _gold_identities(annotations, document_names=artifact_documents)
    decisions = _identity_decisions(artifacts)
    gold_keys = set(gold)
    decision_keys = set(decisions)
    correct_gold = {key for key, label in gold.items() if label == "CORRECT_IDENTITY"}
    resolved = {key for key, outcome in decisions.items() if outcome == "resolved"}
    resolved_labeled = resolved & gold_keys
    matrix: Counter[tuple[str, str]] = Counter(
        (decisions.get(key, "not_reached"), label) for key, label in gold.items()
    )
    incorrect_admissions = [
        {
            "document": key[0],
            "locator_span": {"start": key[1], "end": key[2]},
            "outcome": decisions[key],
            "gold_label": gold[key],
        }
        for key in resolved & {key for key, label in gold.items() if label == "WRONG_IDENTITY"}
    ]
    incorrect_admissions.sort(key=lambda row: (str(row["document"]), row["locator_span"]["start"]))
    outcome_by_gold = {
        outcome: {
            "correct_identity": matrix[(outcome, "CORRECT_IDENTITY")],
            "wrong_identity": matrix[(outcome, "WRONG_IDENTITY")],
        }
        for outcome in sorted({outcome for outcome, _label in matrix})
    }
    return {
        "artifact_type": "full_reporter_locator_identity_score",
        "annotations": str(annotations),
        "artifacts": str(artifacts),
        "labeled_roots": len(gold),
        "stage_coverage": {
            "gold": len(gold_keys),
            "reached": len(gold_keys & decision_keys),
            "not_reached": len(gold_keys - decision_keys),
        },
        "admitted_correct_identity": score_sets(correct_gold, resolved_labeled),
        "outcome_by_gold_label": outcome_by_gold,
        "unlabeled_resolutions": len(resolved - gold_keys),
        "incorrect_admissions": incorrect_admissions,
    }


def _gold_identities(
    annotations: Path,
    *,
    document_names: set[str],
) -> dict[tuple[str, int, int], str]:
    """Read labeled full-reporter root identities by document and locator span."""
    gold: dict[tuple[str, int, int], str] = {}
    for path in sorted(annotations.glob("*.jsonl")):
        if path.stem not in document_names:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("unit") != "citation" or row.get("kind") != "FullCaseCitation":
                continue
            identity = row.get("validation", {}).get("identity", {})
            label = identity.get("label") if isinstance(identity, dict) else None
            locator = row.get("locator")
            if label not in _IDENTITY_LABELS or not isinstance(locator, dict):
                continue
            key = (path.stem, int(locator["start"]), int(locator["end"]))
            if key in gold:
                raise ValueError(f"Duplicate annotated full-reporter root: {key}")
            gold[key] = label
    return gold


def _identity_decisions(artifacts: Path) -> dict[tuple[str, int, int], str]:
    """Read terminal root-identity outcomes by document and locator span."""
    decisions: dict[tuple[str, int, int], str] = {}
    for path in sorted(artifacts.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for citation in payload.get("citations", []):
            source = citation.get("source", {})
            if source.get("citation_type") != "FullCaseCitation":
                continue
            outcome = _resolution_outcome(citation)
            locator = source.get("locator_span")
            if outcome is None or not isinstance(locator, dict):
                continue
            key = (path.stem, int(locator["start"]), int(locator["end"]))
            if key in decisions:
                raise ValueError(f"Duplicate root identity decision: {key}")
            decisions[key] = outcome
    return decisions


def _resolution_outcome(citation: dict[str, Any]) -> str | None:
    """Find this citation's explicit terminal identity decision in its trace."""
    for node in citation.get("trace", []):
        details = node.get("details", {})
        if details.get("validation_node_type") != "LocatorIdentityResolutionNode":
            continue
        validation = details.get("validation")
        outcome = validation.get("outcome") if isinstance(validation, dict) else None
        if isinstance(outcome, str):
            return outcome
    return None


def main() -> None:
    """Write or print a re-scoreable root-identity score artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score_full_reporter_locator_identity(annotations=args.annotations, artifacts=args.artifacts)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
