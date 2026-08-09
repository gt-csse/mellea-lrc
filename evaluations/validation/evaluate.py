"""Evaluate CourtListener identity verdicts against a frozen JSONL bench."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

VERDICTS = frozenset({"match", "mismatch"})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSON object per nonblank line."""
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except json.JSONDecodeError as error:
        msg = f"{path}: invalid JSONL: {error}"
        raise ValueError(msg) from error


def _require_occurrence(record: dict[str, Any], *, source: str, require_verdict: bool) -> None:
    """Validate the stable occurrence fields shared by bench and run artifacts."""
    if not isinstance(record.get("id"), str) or not record["id"]:
        raise ValueError(f"{source}: missing non-empty id")
    locator_id = record.get("locator_id")
    if locator_id is not None and not isinstance(locator_id, str):
        raise ValueError(f"{source} {record['id']}: locator_id must be a string or null")
    span = record.get("locator_span")
    if (
        not isinstance(span, dict)
        or not isinstance(span.get("start"), int)
        or not isinstance(span.get("end"), int)
    ):
        raise ValueError(f"{source} {record['id']}: locator_span must contain integer start and end")
    if span["start"] < 0 or span["end"] <= span["start"]:
        raise ValueError(f"{source} {record['id']}: locator_span must be a non-empty half-open span")
    if require_verdict and not isinstance(record.get("verdict"), str):
        raise ValueError(f"{source} {record['id']}: missing verdict")


def _by_id(records: list[dict[str, Any]], *, source: str) -> dict[str, dict[str, Any]]:
    """Index records and reject duplicate occurrence IDs."""
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = record["id"]
        if record_id in indexed:
            raise ValueError(f"{source}: duplicate id {record_id}")
        indexed[record_id] = record
    return indexed


def _same_occurrence(benchmark: dict[str, Any], artifact: dict[str, Any]) -> bool:
    """Return whether two records describe the same extracted locator occurrence."""
    return (
        benchmark["locator_id"] == artifact["locator_id"]
        and benchmark["locator_span"] == artifact["locator_span"]
    )


def evaluate(
    benchmark_path: Path, artifact_path: Path
) -> tuple[Counter[tuple[str, str]], list[dict[str, Any]]]:
    """Evaluate one run artifact and return the matrix and detailed disagreements."""
    benchmark_rows = _read_jsonl(benchmark_path)
    artifact_rows = _read_jsonl(artifact_path)
    for row in benchmark_rows:
        _require_occurrence(row, source="benchmark", require_verdict=False)
    for row in artifact_rows:
        _require_occurrence(row, source="artifact", require_verdict=True)

    benchmark = _by_id(benchmark_rows, source="benchmark")
    artifact = _by_id(artifact_rows, source="artifact")
    # Every benchmark row carries a verdict: the bench holds only occurrences
    # CourtListener can decide. Whether an occurrence is reachable at all is an
    # extraction question, evaluated separately.
    labeled = {
        record_id: row for record_id, row in benchmark.items() if row.get("expected_verdict") in VERDICTS
    }
    if not labeled:
        raise ValueError("benchmark has no rows with an expected match or mismatch verdict")

    matrix: Counter[tuple[str, str]] = Counter()
    disagreements: list[dict[str, Any]] = []
    for record_id, expected in labeled.items():
        observed = artifact.get(record_id)
        if observed is None:
            disagreements.append(
                {"reason": "missing_artifact_record", "expected": expected, "observed": None}
            )
            continue
        if not _same_occurrence(expected, observed):
            raise ValueError(f"artifact {record_id}: locator_id or locator_span differs from benchmark")
        verdict = observed["verdict"]
        matrix[(verdict, expected["expected_verdict"])] += 1
        if verdict not in VERDICTS:
            reason = "non_confident_verdict"
        elif verdict != expected["expected_verdict"]:
            reason = "verdict_disagreement"
        else:
            continue
        disagreements.append({"reason": reason, "expected": expected, "observed": observed})
    return matrix, disagreements


def _format_matrix(matrix: Counter[tuple[str, str]], disagreement_count: int) -> str:
    """Render a compact, conventional confusion matrix."""
    rows = [
        "| Observed verdict | Expected `match` | Expected `mismatch` |",
        "|---|---:|---:|",
    ]
    observed_verdicts = [
        "match",
        "mismatch",
        *(sorted({verdict for verdict, _ in matrix if verdict not in VERDICTS})),
    ]
    for observed in observed_verdicts:
        rows.append(f"| `{observed}` | {matrix[(observed, 'match')]} | {matrix[(observed, 'mismatch')]} |")
    evaluated = sum(matrix.values())
    rows.append(f"\nEvaluated labeled records: {evaluated}. Non-agreements: {disagreement_count}.")
    return "\n".join(rows)


def main() -> None:
    """Run the command-line evaluator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True, help="Validation bench JSONL file.")
    parser.add_argument("--artifact", type=Path, required=True, help="One JSONL run artifact.")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation-result"))
    args = parser.parse_args()

    matrix, disagreements = evaluate(args.benchmark, args.artifact)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "non_agreements.json"
    output.write_text(json.dumps(disagreements, indent=2) + "\n", encoding="utf-8")
    print(_format_matrix(matrix, len(disagreements)))
    print(f"Details: {output}")


if __name__ == "__main__":
    main()
