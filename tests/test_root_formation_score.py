"""Tests for root formation's independent graph evaluator."""

from __future__ import annotations

import json
from pathlib import Path

from evaluations.extraction.score_root_formation import score_root_formation


def test_score_separates_root_locator_detection_from_attribution(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    artifacts = tmp_path / "artifacts"
    annotations.mkdir()
    artifacts.mkdir()
    _write_jsonl(
        annotations / "001.jsonl",
        [
            _annotation("r1", "r1", "FullCaseCitation", 10, 20),
            _annotation("r2", "r1", "FullCaseCitation", 30, 40),
            _annotation("d1", "d1", "DocketCitation", 50, 60),
        ],
    )
    _write_json(
        artifacts / "001.json",
        {
            "citations": [
                _citation("p1", "p1", "FullCaseCitation", 10, 20),
                _citation("p2", "p1", "FullCaseCitation", 30, 40),
                _citation("p3", "p3", "DocketCitation", 50, 60),
                _citation("p4", "p4", "FullCaseCitation", 70, 80),
            ]
        },
    )

    score = score_root_formation(annotations=annotations, artifacts=artifacts)

    assert score["root_locators"] == {
        "gold": 2,
        "predicted": 3,
        "tp": 2,
        "fp": 1,
        "fn": 0,
        "precision": 2 / 3,
        "recall": 1.0,
        "f1": 0.8,
    }
    assert score["locator_attribution"] == {
        "gold": 3,
        "predicted": 4,
        "tp": 3,
        "fp": 1,
        "fn": 0,
        "precision": 0.75,
        "recall": 1.0,
        "f1": 0.8571428571428571,
    }


def _annotation(row_id: str, root_id: str, kind: str, start: int, end: int) -> dict[str, object]:
    return {
        "unit": "citation",
        "id": row_id,
        "root_id": root_id,
        "kind": kind,
        "locator": {"start": start, "end": end},
    }


def _citation(citation_id: str, root_id: str, kind: str, start: int, end: int) -> dict[str, object]:
    return {
        "citation_id": citation_id,
        "root_id": root_id,
        "source": {"citation_type": kind, "locator_span": {"start": start, "end": end}},
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
