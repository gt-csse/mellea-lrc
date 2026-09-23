"""Score the bounded exact-lookup ambiguity stage from persisted Documents.

An ambiguous root is defined by the saved ``ExactLocatorLookupNode`` candidate
set, independent of how its later identity resolution ends.  This keeps the
metric narrow: it measures whether the ambiguity stage admits correct roots
and rejects wrong ones, rather than conflating it with exact lookup coverage.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluations.extraction.locator_eval_common import score_sets
from evaluations.validation.score_full_reporter_locator_identity import _gold_identities


def score_full_reporter_locator_ambiguity(*, annotations: Path, artifacts: Path) -> dict[str, object]:
    """Score final identity outcomes only for roots with an ambiguous exact lookup."""
    document_names = {path.stem for path in artifacts.glob("*.json")}
    gold = _gold_identities(annotations, document_names=document_names)
    candidates, candidate_counts, decisions = _ambiguous_records(artifacts)
    evaluated_gold = {key: gold[key] for key in candidates & set(gold)}
    correct_gold = {key for key, label in evaluated_gold.items() if label == "CORRECT_IDENTITY"}
    resolved = {key for key, outcome in decisions.items() if outcome == "resolved"}
    matrix: Counter[tuple[str, str]] = Counter(
        (decisions.get(key, "not_reached"), label) for key, label in evaluated_gold.items()
    )
    outcome_by_gold = {
        outcome: {
            "correct_identity": matrix[(outcome, "CORRECT_IDENTITY")],
            "wrong_identity": matrix[(outcome, "WRONG_IDENTITY")],
        }
        for outcome in sorted({outcome for outcome, _label in matrix})
    }
    incorrect_admissions = [
        {
            "document": key[0],
            "locator_span": {"start": key[1], "end": key[2]},
            "outcome": decisions[key],
            "gold_label": evaluated_gold[key],
        }
        for key in resolved
        if evaluated_gold.get(key) == "WRONG_IDENTITY"
    ]
    incorrect_admissions.sort(key=lambda row: (str(row["document"]), row["locator_span"]["start"]))
    return {
        "artifact_type": "full_reporter_locator_ambiguity_score",
        "annotations": str(annotations),
        "artifacts": str(artifacts),
        "ambiguous_roots": len(candidates),
        "labeled_ambiguous_roots": len(evaluated_gold),
        "candidate_count_distribution": dict(sorted(Counter(candidate_counts.values()).items())),
        "stage_coverage": {
            "ambiguous_roots": len(candidates),
            "labeled": len(evaluated_gold),
            "unlabeled": len(candidates - set(gold)),
            "identity_decision": len(decisions),
            "not_reached": len(evaluated_gold) - len(decisions),
        },
        "admitted_correct_identity": score_sets(correct_gold, resolved & set(evaluated_gold)),
        "outcome_by_gold_label": outcome_by_gold,
        "incorrect_admissions": incorrect_admissions,
    }


def _ambiguous_records(
    artifacts: Path,
) -> tuple[set[tuple[str, int, int]], dict[tuple[str, int, int], int], dict[tuple[str, int, int], str]]:
    """Read the exact candidate set and final identity decision for each ambiguous root."""
    candidates: set[tuple[str, int, int]] = set()
    candidate_counts: dict[tuple[str, int, int], int] = {}
    decisions: dict[tuple[str, int, int], str] = {}
    for path in sorted(artifacts.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for citation in payload.get("citations", []):
            source = citation.get("fields")
            if (
                not isinstance(source, dict)
                or source.get("kind") != "FullCaseCitation"
                or citation.get("root_id") != citation.get("citation_id")
            ):
                continue
            locator = source.get("locator_span")
            if not isinstance(locator, dict):
                continue
            lookup = _validation_node(citation, "ExactLocatorLookupNode")
            if lookup is None or lookup.get("outcome") != "ambiguous":
                continue
            candidate_count = lookup.get("candidate_count")
            if not isinstance(candidate_count, int):
                msg = f"{path}: ambiguous lookup has no integer candidate_count"
                raise ValueError(msg)
            key = (path.stem, int(locator["start"]), int(locator["end"]))
            if key in candidates:
                raise ValueError(f"Duplicate ambiguous root: {key}")
            candidates.add(key)
            candidate_counts[key] = candidate_count
            resolution = _validation_node(citation, "LocatorIdentityResolutionNode")
            if resolution is not None and isinstance(resolution.get("outcome"), str):
                decisions[key] = str(resolution["outcome"])
    return candidates, candidate_counts, decisions


def _validation_node(citation: dict[str, Any], node_type: str) -> dict[str, Any] | None:
    """Find one persisted typed validation node in a stage-neutral trace."""
    for trace_node in citation.get("trace", []):
        details = trace_node.get("details")
        if not isinstance(details, dict) or details.get("validation_node_type") != node_type:
            continue
        payload = details.get("validation")
        if isinstance(payload, dict):
            return payload
    return None


def main() -> None:
    """Write or print a re-scoreable ambiguity-stage score artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score_full_reporter_locator_ambiguity(annotations=args.annotations, artifacts=args.artifacts)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
