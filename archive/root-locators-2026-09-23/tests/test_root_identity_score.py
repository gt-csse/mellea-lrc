"""Tests for the final, route-neutral root-identity metric."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluations.validation.score_root_identity import score_root_identity


def test_root_identity_score_separates_admission_and_two_sided_decision(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    artifacts = tmp_path / "artifacts"
    annotations.mkdir()
    artifacts.mkdir()
    _write_jsonl(
        annotations / "001__example.jsonl",
        [
            {"unit": "header"},
            _annotation("FullCaseCitation", 10, 20, "CORRECT_IDENTITY"),
            _annotation("DocketCitation", 30, 40, "WRONG_IDENTITY"),
            _annotation("FullCaseCitation", 50, 60, "CORRECT_IDENTITY"),
        ],
    )
    (artifacts / "001__example.json").write_text(
        json.dumps(
            {
                "citations": [
                    _citation("FullCaseCitation", 10, 20, "resolved"),
                    _citation("DocketCitation", 30, 40, "no_match"),
                    _citation("FullCaseCitation", 50, 60, "deferred_to_open_web_search"),
                ]
            }
        ),
        encoding="utf-8",
    )

    score = score_root_identity(annotations=annotations, artifacts=artifacts)
    all_root = score["scores"]["all_root"]
    docket_root = score["scores"]["docket_root"]
    reporter_root = score["scores"]["reporter_root"]

    assert all_root["positive_admission"] == {
        "gold": 2,
        "predicted": 1,
        "tp": 1,
        "fp": 0,
        "fn": 1,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 2 / 3,
    }
    assert all_root["decision"] == {
        "decided": 2,
        "deferred_or_unreached": 1,
        "correct": 2,
        "accuracy": 1.0,
    }
    assert all_root["root_identity_recall"] == {
        "gold": 3,
        "correct_admissions": 1,
        "correct_rejections": 1,
        "correct": 2,
        "not_correct": 1,
        "recall": 2 / 3,
    }
    assert docket_root["root_identity_recall"] == {
        "gold": 1,
        "correct_admissions": 0,
        "correct_rejections": 1,
        "correct": 1,
        "not_correct": 0,
        "recall": 1.0,
    }
    assert reporter_root["root_identity_recall"] == {
        "gold": 2,
        "correct_admissions": 1,
        "correct_rejections": 0,
        "correct": 1,
        "not_correct": 1,
        "recall": 0.5,
    }
    assert all_root["root_identity_recall"]["gold"] == (
        docket_root["root_identity_recall"]["gold"] + reporter_root["root_identity_recall"]["gold"]
    )
    assert all_root["root_identity_recall"]["correct"] == (
        docket_root["root_identity_recall"]["correct"] + reporter_root["root_identity_recall"]["correct"]
    )


def test_root_identity_recall_counts_wrong_decisions_and_missing_documents(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    artifacts = tmp_path / "artifacts"
    annotations.mkdir()
    artifacts.mkdir()
    _write_jsonl(
        annotations / "001__example.jsonl",
        [
            _annotation("FullCaseCitation", 10, 20, "CORRECT_IDENTITY"),
            _annotation("DocketCitation", 30, 40, "WRONG_IDENTITY"),
        ],
    )
    _write_jsonl(
        annotations / "002__missing.jsonl",
        [_annotation("FullCaseCitation", 50, 60, "CORRECT_IDENTITY")],
    )
    (artifacts / "001__example.json").write_text(
        json.dumps(
            {
                "citations": [
                    _citation("FullCaseCitation", 10, 20, "resolved"),
                    _citation("DocketCitation", 30, 40, "resolved"),
                ]
            }
        ),
        encoding="utf-8",
    )

    score = score_root_identity(annotations=annotations, artifacts=artifacts)

    assert score["scores"]["all_root"]["root_identity_recall"] == {
        "gold": 3,
        "correct_admissions": 1,
        "correct_rejections": 0,
        "correct": 1,
        "not_correct": 2,
        "recall": 1 / 3,
    }
    assert score["scores"]["all_root"]["stage_coverage"] == {
        "gold": 3,
        "reached": 2,
        "not_reached": 1,
        "missing_documents": ["002__missing"],
    }
    assert score["scores"]["all_root"]["positive_admission"]["recall"] == 0.5
    assert score["scores"]["all_root"]["decision"]["accuracy"] == 0.5


def test_root_identity_score_joins_annotation_to_recorded_source_path(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    artifacts = tmp_path / "artifacts"
    annotations.mkdir()
    artifacts.mkdir()
    _write_jsonl(
        annotations / "001__example-.jsonl",
        [
            {"unit": "header", "text": {"path": "corpus/001__example-.txt"}},
            _annotation("FullCaseCitation", 10, 20, "CORRECT_IDENTITY"),
        ],
    )
    # The checkpoint filename lost the hyphen, but the serialized Document
    # retained the original input path.
    (artifacts / "001__example.json").write_text(
        json.dumps(
            {
                "source_metadata": {"path": "corpus/001__example-.txt"},
                "citations": [_citation("FullCaseCitation", 10, 20, "resolved")],
            }
        ),
        encoding="utf-8",
    )

    score = score_root_identity(annotations=annotations, artifacts=artifacts)

    assert score["scores"]["all_root"]["labeled_roots"] == 1
    assert score["scores"]["all_root"]["stage_coverage"] == {
        "gold": 1,
        "reached": 1,
        "not_reached": 0,
        "missing_documents": [],
    }
    assert score["scores"]["all_root"]["positive_admission"]["tp"] == 1
    assert score["scores"]["docket_root"]["root_identity_recall"]["gold"] == 0
    assert score["scores"]["docket_root"]["root_identity_recall"]["recall"] is None
    assert score["scores"]["reporter_root"]["root_identity_recall"]["gold"] == 1


def test_root_identity_score_rejects_duplicate_source_document_mapping(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    artifacts = tmp_path / "artifacts"
    annotations.mkdir()
    artifacts.mkdir()
    payload = {"source_metadata": {"path": "corpus/001__example-.txt"}, "citations": []}
    (artifacts / "first.json").write_text(json.dumps(payload), encoding="utf-8")
    (artifacts / "second.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Multiple artifacts map to source document"):
        score_root_identity(annotations=annotations, artifacts=artifacts)


def test_root_identity_score_rejects_legacy_checkpoint(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    artifacts = tmp_path / "artifacts"
    annotations.mkdir()
    artifacts.mkdir()
    _write_jsonl(
        annotations / "001__example.jsonl",
        [_annotation("FullCaseCitation", 10, 20, "CORRECT_IDENTITY")],
    )
    (artifacts / "001__example.json").write_text(
        json.dumps(
            {
                "citations": [
                    {
                        "citation_id": "root",
                        "root_id": "root",
                        "source": {
                            "citation_type": "FullCaseCitation",
                            "locator_span": {"start": 10, "end": 20},
                        },
                        "judgements": {"identity": {"outcome": "resolved"}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported legacy Document checkpoint"):
        score_root_identity(annotations=annotations, artifacts=artifacts)


def test_root_identity_score_rejects_unpartitioned_gold_kind(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    artifacts = tmp_path / "artifacts"
    annotations.mkdir()
    artifacts.mkdir()
    _write_jsonl(
        annotations / "001__example.jsonl",
        [_annotation("OtherCitation", 10, 20, "CORRECT_IDENTITY")],
    )
    (artifacts / "001__example.json").write_text(json.dumps({"citations": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="no identity partition"):
        score_root_identity(annotations=annotations, artifacts=artifacts)


def test_root_identity_score_marks_unlabeled_set_not_evaluable(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    artifacts = tmp_path / "artifacts"
    annotations.mkdir()
    artifacts.mkdir()
    _write_jsonl(annotations / "001__example.jsonl", [{"unit": "header"}])
    (artifacts / "001__example.json").write_text(json.dumps({"citations": []}), encoding="utf-8")

    score = score_root_identity(annotations=annotations, artifacts=artifacts)

    assert tuple(score["scores"]) == ("all_root", "docket_root", "reporter_root")
    for partition in score["scores"].values():
        assert partition["root_identity_recall"]["gold"] == 0
        assert partition["root_identity_recall"]["recall"] is None


def _annotation(kind: str, start: int, end: int, label: str) -> dict[str, object]:
    return {
        "unit": "citation",
        "is_root": True,
        "kind": kind,
        "locator": {"start": start, "end": end},
        "validation": {"identity": {"label": label}},
    }


def _citation(kind: str, start: int, end: int, outcome: str) -> dict[str, object]:
    return {
        "citation_id": f"{kind}-{start}",
        "root_id": f"{kind}-{start}",
        "fields": {"kind": kind, "locator_span": {"start": start, "end": end}},
        "judgements": {"identity": {"outcome": outcome}},
    }


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")
