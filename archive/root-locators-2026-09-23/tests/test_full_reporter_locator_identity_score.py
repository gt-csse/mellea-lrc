"""Tests for the persisted root-identity evaluation boundary."""

from __future__ import annotations

import json
from pathlib import Path

from evaluations.validation.score_full_reporter_locator_identity import (
    score_full_reporter_locator_identity,
)


def test_score_uses_labeled_roots_in_completed_artifact_documents(tmp_path: Path) -> None:
    """Unlabeled resolutions and unprocessed annotation files stay out of accuracy."""
    annotations = tmp_path / "annotations"
    artifacts = tmp_path / "artifacts"
    annotations.mkdir()
    artifacts.mkdir()
    _write_jsonl(
        annotations / "001.jsonl",
        [
            _annotation(start=10, end=20, label="CORRECT_IDENTITY"),
            _annotation(start=30, end=40, label="WRONG_IDENTITY"),
            _annotation(start=90, end=100, label="CORRECT_IDENTITY", is_root=False),
        ],
    )
    _write_jsonl(
        annotations / "002.jsonl",
        [_annotation(start=50, end=60, label="CORRECT_IDENTITY")],
    )
    _write_json(
        artifacts / "001.json",
        {
            "citations": [
                _citation(start=10, end=20, outcome="resolved"),
                _citation(start=30, end=40, outcome="deferred_to_search"),
                _citation(start=70, end=80, outcome="resolved"),
            ]
        },
    )

    result = score_full_reporter_locator_identity(annotations=annotations, artifacts=artifacts)

    assert result["labeled_roots"] == 2
    assert result["stage_coverage"] == {"gold": 2, "reached": 2, "not_reached": 0}
    assert result["admitted_correct_identity"] == {
        "gold": 1,
        "predicted": 1,
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert result["outcome_by_gold_label"] == {
        "deferred_to_search": {"correct_identity": 0, "wrong_identity": 1},
        "resolved": {"correct_identity": 1, "wrong_identity": 0},
    }
    assert result["unlabeled_resolutions"] == 1


def test_score_uses_the_current_judgment_after_an_earlier_deferred_resolution(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    artifacts = tmp_path / "artifacts"
    annotations.mkdir()
    artifacts.mkdir()
    _write_jsonl(annotations / "001.jsonl", [_annotation(start=10, end=20, label="CORRECT_IDENTITY")])
    citation = _citation(start=10, end=20, outcome="deferred_to_search")
    citation["trace"].append(
        {
            "details": {
                "validation_node_type": "LocatorIdentityResolutionNode",
                "validation": {"outcome": "resolved"},
            }
        }
    )
    citation["judgements"] = {"identity": {"outcome": "resolved"}}
    _write_json(artifacts / "001.json", {"citations": [citation]})

    result = score_full_reporter_locator_identity(annotations=annotations, artifacts=artifacts)

    assert result["admitted_correct_identity"]["tp"] == 1
    assert result["outcome_by_gold_label"] == {"resolved": {"correct_identity": 1, "wrong_identity": 0}}


def _annotation(*, start: int, end: int, label: str, is_root: bool = True) -> dict[str, object]:
    return {
        "unit": "citation",
        "is_root": is_root,
        "kind": "FullCaseCitation",
        "locator": {"start": start, "end": end},
        "validation": {"identity": {"label": label}},
    }


def _citation(*, start: int, end: int, outcome: str) -> dict[str, object]:
    return {
        "citation_id": f"citation-{start}",
        "root_id": f"citation-{start}",
        "fields": {
            "kind": "FullCaseCitation",
            "locator_span": {"start": start, "end": end},
        },
        "trace": [
            {
                "details": {
                    "validation_node_type": "LocatorIdentityResolutionNode",
                    "validation": {"outcome": outcome},
                }
            }
        ],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
