"""Score persisted docket-root identity decisions.

The evaluator scores only roots with an explicit annotation identity label. A
``resolved`` decision is an admission; every other terminal state remains in
the outcome-by-gold matrix for diagnosis.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluations.extraction.locator_eval_common import score_sets

_IDENTITY_LABELS = frozenset({"CORRECT_IDENTITY", "WRONG_IDENTITY"})


def score_docket_root_identity(*, annotations: Path, artifacts: Path) -> dict[str, object]:
    """Score one completed docket-root identity artifact directory."""
    artifact_documents = {path.stem for path in artifacts.glob("*.json")}
    gold = _gold_identities(annotations, document_names=artifact_documents)
    decisions = _identity_decisions(artifacts)
    correct_gold = {key for key, label in gold.items() if label == "CORRECT_IDENTITY"}
    resolved = {key for key, outcome in decisions.items() if outcome == "resolved"}
    resolved_labeled = resolved & set(gold)
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
    return {
        "artifact_type": "docket_root_identity_score",
        "annotations": str(annotations),
        "artifacts": str(artifacts),
        "labeled_roots": len(gold),
        "stage_coverage": {
            "gold": len(gold),
            "reached": len(set(gold) & set(decisions)),
            "not_reached": len(set(gold) - set(decisions)),
        },
        "admitted_correct_identity": score_sets(correct_gold, resolved_labeled),
        "outcome_by_gold_label": {
            outcome: {
                "correct_identity": matrix[(outcome, "CORRECT_IDENTITY")],
                "wrong_identity": matrix[(outcome, "WRONG_IDENTITY")],
            }
            for outcome in sorted({outcome for outcome, _label in matrix})
        },
        "unlabeled_resolutions": len(resolved - set(gold)),
        "incorrect_admissions": sorted(
            incorrect_admissions,
            key=lambda row: (str(row["document"]), int(row["locator_span"]["start"])),
        ),
    }


def _gold_identities(
    annotations: Path,
    *,
    document_names: set[str],
) -> dict[tuple[str, int, int], str]:
    gold: dict[tuple[str, int, int], str] = {}
    for path in sorted(annotations.glob("*.jsonl")):
        if path.stem not in document_names:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if (
                row.get("unit") != "citation"
                or row.get("kind") != "DocketCitation"
                or row.get("is_root") is not True
            ):
                continue
            identity = row.get("validation", {}).get("identity", {})
            label = identity.get("label") if isinstance(identity, dict) else None
            locator = row.get("locator")
            if label not in _IDENTITY_LABELS or not isinstance(locator, dict):
                continue
            key = (path.stem, int(locator["start"]), int(locator["end"]))
            if key in gold:
                raise ValueError(f"Duplicate annotated docket root: {key}")
            gold[key] = label
    return gold


def _identity_decisions(artifacts: Path) -> dict[tuple[str, int, int], str]:
    decisions: dict[tuple[str, int, int], str] = {}
    for path in sorted(artifacts.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for citation in payload.get("citations", []):
            source = citation.get("source", {})
            if source.get("citation_type") != "DocketCitation" or citation.get("root_id") != citation.get(
                "citation_id"
            ):
                continue
            judgement = citation.get("judgements", {}).get("identity", {})
            outcome = judgement.get("outcome") if isinstance(judgement, dict) else None
            locator = source.get("locator_span")
            if not isinstance(outcome, str) or not isinstance(locator, dict):
                continue
            key = (path.stem, int(locator["start"]), int(locator["end"]))
            if key in decisions:
                raise ValueError(f"Duplicate docket root identity decision: {key}")
            decisions[key] = outcome
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score_docket_root_identity(annotations=args.annotations, artifacts=args.artifacts)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
