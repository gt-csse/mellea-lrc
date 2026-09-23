"""Score final root-identity judgements against citation annotations.

This evaluator is intentionally neutral about retrieval routes.  It reads the
current first-class ``identity`` judgement from each serialized ``Document``:
``resolved`` is a positive identity admission, ``no_match`` is a negative
identity decision, and every deferred status remains explicitly unscored.

The same measures are reported for ``all_root``, ``docket_root``, and
``reporter_root``. Root-identity recall is (correct ``resolved`` + correct
``no_match``) divided by every identity-labeled root in that partition.
Missing Documents, missing roots, and deferred outcomes remain in the
denominator. Positive-admission recall is a separate diagnostic measure over
gold-correct roots in the same partition.

Example:
    uv run python -m evaluations.validation.score_root_identity \\
      --annotations data/primary/documents \\
      --artifacts data/run-artifacts/RUN/shared_body_evidence_review \\
      --output evaluations/validation/reports/RUN-root-identity.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluations.extraction.locator_eval_common import score_sets

_LABELS = frozenset({"CORRECT_IDENTITY", "WRONG_IDENTITY"})
_DECISIONS = frozenset({"resolved", "no_match"})
_ROOT_PARTITIONS = {
    "docket_root": "DocketCitation",
    "reporter_root": "FullCaseCitation",
}


def score_root_identity(*, annotations: Path, artifacts: Path) -> dict[str, object]:
    """Score all roots and both disjoint root kinds from one saved checkpoint."""
    outcomes, artifact_documents = _root_outcomes(artifacts)
    gold = _gold_roots(annotations)
    unknown_kinds = sorted({key[1] for key in gold} - set(_ROOT_PARTITIONS.values()))
    if unknown_kinds:
        raise ValueError(f"Annotated root kinds have no identity partition: {unknown_kinds}")
    partitions = {
        "all_root": gold,
        **{
            name: {key: label for key, label in gold.items() if key[1] == kind}
            for name, kind in _ROOT_PARTITIONS.items()
        },
    }
    if len(partitions["all_root"]) != len(partitions["docket_root"]) + len(partitions["reporter_root"]):
        raise ValueError("Docket and reporter partitions do not cover all annotated roots")
    return {
        "artifact_type": "root_identity_score",
        "schema_version": 2,
        "annotations": str(annotations),
        "artifacts": str(artifacts),
        "scores": {
            name: _score_partition(partition, outcomes, artifact_documents)
            for name, partition in partitions.items()
        },
    }


def _score_partition(
    gold: dict[tuple[str, str, int, int], str],
    outcomes: dict[tuple[str, str, int, int], str],
    artifact_documents: set[str],
) -> dict[str, object]:
    """Apply identical scoring semantics to every root partition."""
    gold_keys = set(gold)
    reached = gold_keys & set(outcomes)
    positive_gold = {key for key, label in gold.items() if label == "CORRECT_IDENTITY"}
    positive_admissions = {key for key, outcome in outcomes.items() if outcome == "resolved"}
    matrix: Counter[tuple[str, str]] = Counter(
        (outcomes.get(key, "not_reached"), label) for key, label in gold.items()
    )
    decided = {key: outcome for key, outcome in outcomes.items() if key in gold and outcome in _DECISIONS}
    correct_decisions = {
        key
        for key, outcome in decided.items()
        if (outcome == "resolved" and gold[key] == "CORRECT_IDENTITY")
        or (outcome == "no_match" and gold[key] == "WRONG_IDENTITY")
    }
    correct_admissions = sum(outcomes[key] == "resolved" for key in correct_decisions)
    correct_rejections = len(correct_decisions) - correct_admissions
    wrong_admissions = sorted(
        (
            _row(key, outcome, gold[key])
            for key, outcome in outcomes.items()
            if outcome == "resolved" and gold.get(key) == "WRONG_IDENTITY"
        ),
        key=_row_sort_key,
    )
    wrong_rejections = sorted(
        (
            _row(key, outcome, gold[key])
            for key, outcome in outcomes.items()
            if outcome == "no_match" and gold.get(key) == "CORRECT_IDENTITY"
        ),
        key=_row_sort_key,
    )
    return {
        "labeled_roots": len(gold),
        "root_identity_recall": {
            "gold": len(gold),
            "correct_admissions": correct_admissions,
            "correct_rejections": correct_rejections,
            "correct": len(correct_decisions),
            "not_correct": len(gold) - len(correct_decisions),
            "recall": len(correct_decisions) / len(gold) if gold else None,
        },
        "stage_coverage": {
            "gold": len(gold),
            "reached": len(reached),
            "not_reached": len(gold_keys - reached),
            "missing_documents": sorted({key[0] for key in gold_keys} - artifact_documents),
        },
        "positive_admission": score_sets(positive_gold, positive_admissions & gold_keys),
        "decision": {
            "decided": len(decided),
            "deferred_or_unreached": len(gold) - len(decided),
            "correct": len(correct_decisions),
            "accuracy": len(correct_decisions) / len(decided) if decided else None,
        },
        "outcome_by_gold_label": {
            outcome: {
                "correct_identity": matrix[(outcome, "CORRECT_IDENTITY")],
                "wrong_identity": matrix[(outcome, "WRONG_IDENTITY")],
            }
            for outcome in sorted({outcome for outcome, _label in matrix})
        },
        "incorrect_admissions": wrong_admissions,
        "incorrect_rejections": wrong_rejections,
    }


def _gold_roots(
    annotations: Path, *, document_names: set[str] | None = None
) -> dict[tuple[str, str, int, int], str]:
    if not annotations.is_dir():
        raise ValueError(f"Annotation directory does not exist: {annotations}")
    gold: dict[tuple[str, str, int, int], str] = {}
    paths = sorted(annotations.glob("*.jsonl"))
    if not paths:
        raise ValueError(f"No annotation JSONL files in {annotations}")
    for path in paths:
        if document_names is not None and path.stem not in document_names:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("unit") != "citation" or row.get("is_root") is not True:
                continue
            identity = row.get("validation", {}).get("identity", {})
            locator = row.get("locator")
            label = identity.get("label") if isinstance(identity, dict) else None
            kind = row.get("kind")
            if label not in _LABELS or not isinstance(locator, dict) or not isinstance(kind, str):
                continue
            key = (path.stem, kind, int(locator["start"]), int(locator["end"]))
            if key in gold:
                raise ValueError(f"Duplicate annotated root: {key}")
            gold[key] = label
    return gold


def _root_outcomes(
    artifacts: Path,
) -> tuple[dict[tuple[str, str, int, int], str], set[str]]:
    if not artifacts.is_dir():
        raise ValueError(f"Artifact directory does not exist: {artifacts}")
    outcomes: dict[tuple[str, str, int, int], str] = {}
    documents: dict[str, Path] = {}
    for path in sorted(artifacts.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_metadata = payload.get("source_metadata")
        source_path = source_metadata.get("path") if isinstance(source_metadata, dict) else None
        # The checkpoint filename can be normalized independently of the source
        # filename (notably a trailing hyphen before .txt). Match annotations to
        # the explicit source basename when the serialized Document records it.
        document = Path(source_path).stem if isinstance(source_path, str) and source_path else path.stem
        if document in documents:
            raise ValueError(
                f"Multiple artifacts map to source document {document!r}: {documents[document]} and {path}"
            )
        documents[document] = path
        for citation in payload.get("citations", []):
            if citation.get("root_id") != citation.get("citation_id"):
                continue
            if "source" in citation and "fields" not in citation:
                raise ValueError(f"Unsupported legacy Document checkpoint: {path}")
            source = citation.get("fields")
            if not isinstance(source, dict):
                continue
            locator = source.get("locator_span")
            kind = source.get("kind")
            judgement = citation.get("judgements", {}).get("identity", {})
            outcome = judgement.get("outcome") if isinstance(judgement, dict) else None
            if not isinstance(locator, dict) or not isinstance(kind, str) or not isinstance(outcome, str):
                continue
            key = (document, kind, int(locator["start"]), int(locator["end"]))
            if key in outcomes:
                raise ValueError(f"Duplicate serialized root: {key}")
            outcomes[key] = outcome
    return outcomes, set(documents)


def _row(key: tuple[str, str, int, int], outcome: str, label: str) -> dict[str, object]:
    document, kind, start, end = key
    return {
        "document": document,
        "kind": kind,
        "locator_span": {"start": start, "end": end},
        "outcome": outcome,
        "gold_label": label,
    }


def _row_sort_key(row: dict[str, object]) -> tuple[str, int, int]:
    span = row["locator_span"]
    if not isinstance(span, dict):
        raise ValueError("Metric row has no locator span")
    return str(row["document"]), int(span["start"]), int(span["end"])


def main() -> None:
    """Write or print a re-scoreable root-identity metric artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score_root_identity(annotations=args.annotations, artifacts=args.artifacts)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
