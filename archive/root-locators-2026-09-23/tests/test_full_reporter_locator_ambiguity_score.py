"""Contract tests for the narrow exact-lookup ambiguity score."""

from __future__ import annotations

import json
from pathlib import Path

from evaluations.validation.score_full_reporter_locator_ambiguity import (
    score_full_reporter_locator_ambiguity,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_scores_only_ambiguous_full_reporter_roots(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    artifacts = tmp_path / "artifacts"
    annotations.mkdir()
    artifacts.mkdir()
    (annotations / "filing.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "unit": "citation",
                    "kind": "FullCaseCitation",
                    "is_root": True,
                    "locator": {"start": start, "end": end},
                    "validation": {"identity": {"label": label}},
                }
            )
            for start, end, label in (
                (1, 4, "CORRECT_IDENTITY"),
                (10, 13, "WRONG_IDENTITY"),
                (20, 23, "CORRECT_IDENTITY"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        artifacts / "filing.json",
        {
            "citations": [
                _citation(1, 4, "ambiguous", 2, "resolved"),
                _citation(10, 13, "ambiguous", 3, "no_match"),
                _citation(20, 23, "found", 1, "resolved"),
            ]
        },
    )

    result = score_full_reporter_locator_ambiguity(annotations=annotations, artifacts=artifacts)

    assert result["ambiguous_roots"] == 2
    assert result["candidate_count_distribution"] == {2: 1, 3: 1}
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
        "no_match": {"correct_identity": 0, "wrong_identity": 1},
        "resolved": {"correct_identity": 1, "wrong_identity": 0},
    }


def _citation(start: int, end: int, lookup: str, candidate_count: int, resolution: str) -> dict[str, object]:
    return {
        "citation_id": f"citation-{start}",
        "root_id": f"citation-{start}",
        "fields": {"kind": "FullCaseCitation", "locator_span": {"start": start, "end": end}},
        "trace": [
            {
                "details": {
                    "validation_node_type": "ExactLocatorLookupNode",
                    "validation": {"outcome": lookup, "candidate_count": candidate_count},
                }
            },
            {
                "details": {
                    "validation_node_type": "LocatorIdentityResolutionNode",
                    "validation": {"outcome": resolution},
                }
            },
        ],
    }
