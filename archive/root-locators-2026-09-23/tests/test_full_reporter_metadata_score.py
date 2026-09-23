"""Offline score contracts for reporter metadata discovery."""

from __future__ import annotations

import json
from pathlib import Path

from evaluations.validation.score_full_reporter_metadata_search import score_full_reporter_metadata_search


def _citation(*, start: int, end: int, node_type: str, stage: str, candidates: list[dict[str, object]]):
    return {
        "citation_id": "cite-0001",
        "root_id": "cite-0001",
        "fields": {
            "kind": "FullCaseCitation",
            "locator_span": {"start": start, "end": end},
        },
        "trace": [
            {
                "stage": "full_reporter_locator_exact_lookup",
                "details": {
                    "validation_node_type": "ExactLocatorLookupNode",
                    "validation": {"outcome": "not_found"},
                },
            },
            {
                "node_id": f"cite-0001:{stage}:case_name_terms",
                "stage": stage,
                "outcome": "prepared",
                "details": {"terms": ["Example"]},
            },
            {
                "stage": stage,
                "details": {
                    "validation_node_type": node_type,
                    "validation": {
                        "outcome": "found",
                        "reporter_locator": "1 Example 2",
                        "candidate_count": len(candidates),
                        "candidates": candidates,
                        "attempts": [],
                    },
                },
            },
        ],
    }


def _annotation(*, source: dict[str, object], identity_label: str = "CORRECT_IDENTITY"):
    return {
        "unit": "citation",
        "id": "001-o01",
        "kind": "FullCaseCitation",
        "is_root": True,
        "locator": {"start": 12, "end": 23, "quote": "1 Example 2"},
        "validation": {
            "identity": {
                "label": identity_label,
                "evidence": [{"shows": "independent_record", "source": source}],
            }
        },
    }


def test_score_recovers_direct_courtlistener_and_govinfo_docket_targets(tmp_path: Path) -> None:
    annotations = tmp_path / "primary.jsonl"
    annotations.write_text(
        "\n".join(
            json.dumps(value)
            for value in (
                {"unit": "header", "document": "001__example.txt"},
                _annotation(
                    source={
                        "kind": "docket",
                        "id": "44",
                        "external_url": "https://www.courtlistener.com/docket/44/",
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = tmp_path / "courtlistener"
    artifacts.mkdir()
    (artifacts / "001__example.json").write_text(
        json.dumps(
            {
                "citations": [
                    _citation(
                        start=12,
                        end=23,
                        node_type="FullReporterSearchNode",
                        stage="courtlistener_full_reporter_metadata_search",
                        candidates=[{"docket_id": 44}],
                    )
                ]
            }
        ),
        encoding="utf-8",
    )

    score = score_full_reporter_metadata_search(
        annotations=annotations,
        artifacts=artifacts,
        provider="courtlistener",
    )

    assert score["stage_coverage"]["reached"] == 1
    assert score["candidate_retrieval"] == {
        "targets_retrieved": 1,
        "targets_missed": 0,
        "recall": 1.0,
        # The fixture has no provider query attempts. The target was retained
        # in a synthetic node, but it is deliberately outside the distinct
        # queryable-recall denominator.
        "queryable_targets": 0,
        "queryable_targets_retrieved": 0,
        "queryable_recall": None,
    }

    annotations.write_text(
        "\n".join(
            json.dumps(value)
            for value in (
                {"unit": "header", "document": "001__example.txt"},
                _annotation(
                    source={
                        "kind": "docket",
                        "id": "USCOURTS-ncmd-1_18-cv-00754",
                        "external_url": "https://www.govinfo.gov/app/details/USCOURTS-ncmd-1_18-cv-00754",
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    govinfo = tmp_path / "govinfo"
    govinfo.mkdir()
    (govinfo / "001__example.json").write_text(
        json.dumps(
            {
                "citations": [
                    _citation(
                        start=12,
                        end=23,
                        node_type="GovInfoFullReporterSearchNode",
                        stage="govinfo_full_reporter_metadata_search",
                        candidates=[{"govinfo_package_id": "USCOURTS-ncmd-1_18-cv-00754"}],
                    )
                ]
            }
        ),
        encoding="utf-8",
    )

    score = score_full_reporter_metadata_search(
        annotations=annotations,
        artifacts=govinfo,
        provider="govinfo",
    )
    assert score["candidate_retrieval"]["recall"] == 1.0


def test_score_excludes_independent_records_for_wrong_identity(tmp_path: Path) -> None:
    annotations = tmp_path / "primary.jsonl"
    annotations.write_text(
        "\n".join(
            json.dumps(value)
            for value in (
                {"unit": "header", "document": "001__example.txt"},
                _annotation(
                    source={
                        "kind": "docket",
                        "id": "44",
                        "external_url": "https://www.courtlistener.com/docket/44/",
                    },
                    identity_label="WRONG_IDENTITY",
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = tmp_path / "courtlistener"
    artifacts.mkdir()
    (artifacts / "001__example.json").write_text(
        json.dumps(
            {
                "citations": [
                    _citation(
                        start=12,
                        end=23,
                        node_type="FullReporterSearchNode",
                        stage="courtlistener_full_reporter_metadata_search",
                        candidates=[{"docket_id": 44}],
                    )
                ]
            }
        ),
        encoding="utf-8",
    )

    score = score_full_reporter_metadata_search(
        annotations=annotations,
        artifacts=artifacts,
        provider="courtlistener",
    )

    assert score["stage_coverage"]["exact_miss_roots_with_annotated_provider_metadata"] == 0
    assert score["candidate_retrieval"]["recall"] is None


def test_score_excludes_govinfo_opinion_corroboration(tmp_path: Path) -> None:
    annotations = tmp_path / "primary.jsonl"
    annotations.write_text(
        "\n".join(
            json.dumps(value)
            for value in (
                {"unit": "header", "document": "001__example.txt"},
                _annotation(
                    source={
                        "kind": "opinion",
                        "id": "USCOURTS-ncmd-1_18-cv-00754-0",
                        "external_url": "https://www.govinfo.gov/app/details/USCOURTS-ncmd-1_18-cv-00754/USCOURTS-ncmd-1_18-cv-00754-0",
                    },
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = tmp_path / "govinfo"
    artifacts.mkdir()
    (artifacts / "001__example.json").write_text(
        json.dumps(
            {
                "citations": [
                    _citation(
                        start=12,
                        end=23,
                        node_type="GovInfoFullReporterSearchNode",
                        stage="govinfo_full_reporter_metadata_search",
                        candidates=[{"govinfo_package_id": "USCOURTS-ncmd-1_18-cv-00754"}],
                    )
                ]
            }
        ),
        encoding="utf-8",
    )

    score = score_full_reporter_metadata_search(
        annotations=annotations,
        artifacts=artifacts,
        provider="govinfo",
    )

    assert score["stage_coverage"]["exact_miss_roots_with_annotated_provider_metadata"] == 0
