"""Tests for the standalone CourtListener identity evaluator."""

import json
from pathlib import Path

import pytest

from evaluations.validation.evaluate import _format_matrix, evaluate


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_evaluate_reports_confusion_and_non_agreements(tmp_path: Path) -> None:
    benchmark = tmp_path / "bench.jsonl"
    artifact = tmp_path / "artifact.jsonl"
    occurrence = {"locator_id": "fullcasecitation-1-u-s-2", "locator_span": {"start": 10, "end": 18}}
    _write_jsonl(
        benchmark,
        [
            {"id": "one", "covered": True, "expected_verdict": "match", **occurrence},
            {"id": "two", "covered": True, "expected_verdict": "mismatch", **occurrence},
            {"id": "three", "covered": False, **occurrence},
        ],
    )
    _write_jsonl(
        artifact,
        [
            {"id": "one", "verdict": "match", **occurrence},
            {"id": "two", "verdict": "possible_match", **occurrence},
        ],
    )

    matrix, disagreements = evaluate(benchmark, artifact)

    assert matrix[("match", "match")] == 1
    assert matrix[("possible_match", "mismatch")] == 1
    assert disagreements == [
        {
            "reason": "non_confident_verdict",
            "expected": {"id": "two", "covered": True, "expected_verdict": "mismatch", **occurrence},
            "observed": {"id": "two", "verdict": "possible_match", **occurrence},
        }
    ]
    rendered = _format_matrix(matrix, len(disagreements))
    assert rendered.count("| `possible_match` |") == 1


def test_evaluate_rejects_an_artifact_with_a_different_span(tmp_path: Path) -> None:
    benchmark = tmp_path / "bench.jsonl"
    artifact = tmp_path / "artifact.jsonl"
    _write_jsonl(
        benchmark,
        [
            {
                "id": "one",
                "locator_id": "fullcasecitation-1-u-s-2",
                "locator_span": {"start": 10, "end": 18},
                "covered": True,
                "expected_verdict": "match",
            }
        ],
    )
    _write_jsonl(
        artifact,
        [
            {
                "id": "one",
                "locator_id": "fullcasecitation-1-u-s-2",
                "locator_span": {"start": 11, "end": 18},
                "verdict": "match",
            }
        ],
    )

    with pytest.raises(ValueError, match="locator_id or locator_span differs"):
        evaluate(benchmark, artifact)
