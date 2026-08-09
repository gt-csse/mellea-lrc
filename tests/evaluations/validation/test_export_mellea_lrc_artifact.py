"""Tests for flattening Mellea-LRC evaluation artifacts."""

import json
from pathlib import Path

import pytest

from evaluations.validation.export_mellea_lrc_artifact import export


def test_export_flattens_a_serialized_run(tmp_path: Path) -> None:
    artifact = {
        "source": {
            "citations": [
                {
                    "citation_id": "local-id",
                    "locator_span": {"start": 10, "end": 22},
                    "citation": {
                        "citation_type": "FullCaseCitation",
                        "volume": "1",
                        "reporter": "U.S.",
                        "page": "2",
                    },
                }
            ]
        },
        "citations": [{"citation_id": "local-id", "aggregation": {"overall_outcome": "possible_match"}}],
    }
    (tmp_path / "1.json").write_text(json.dumps(artifact), encoding="utf-8")

    assert export(tmp_path) == [
        {
            "id": "cite:1:fullcasecitation-1-u-s-2:10-22",
            "locator_id": "fullcasecitation-1-u-s-2",
            "locator_span": {"start": 10, "end": 22},
            "verdict": "possible_match",
        }
    ]


def test_export_rejects_empty_or_non_numbered_directories(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no serialized JSON files"):
        export(tmp_path)

    (tmp_path / "artifact.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="document number"):
        export(tmp_path)


def test_export_accepts_published_corpus_file_names(tmp_path: Path) -> None:
    """Runs over the published corpus are named 006__slug, not 6.

    The benchmark keys occurrences on the leading document number, so the
    adapter takes that prefix rather than requiring a bare numeric stem.
    """
    artifact = {
        "source": {
            "citations": [
                {
                    "citation_id": "local-id",
                    "locator_span": {"start": 10, "end": 22},
                    "citation": {
                        "citation_type": "FullCaseCitation",
                        "volume": "1",
                        "reporter": "U.S.",
                        "page": "2",
                    },
                }
            ]
        },
        "citations": [{"citation_id": "local-id", "aggregation": {"overall_outcome": "match"}}],
    }
    (tmp_path / "006__coomer-v-lindell-mypillow-inc__response-brief.json").write_text(
        json.dumps(artifact), encoding="utf-8"
    )

    assert [record["id"] for record in export(tmp_path)] == ["cite:006:fullcasecitation-1-u-s-2:10-22"]
