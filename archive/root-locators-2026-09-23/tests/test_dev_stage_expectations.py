from __future__ import annotations

import json
from pathlib import Path

from evaluations.validation.dev_stage_expectations import build_expectations, evaluate_routes


def test_annotation_build_separates_direct_retrieval_from_decision(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    rows = [
        {
            "unit": "citation",
            "id": "d1-o1",
            "is_root": True,
            "kind": "FullCaseCitation",
            "identifier": {"kind": "reporter", "volume": "1", "reporter": "U.S.", "page": "2"},
            "locator": {"start": 1, "end": 5},
            "validation": {
                "identity": {
                    "label": "CORRECT_IDENTITY",
                    "source": "archive",
                    "basis": "record_at_locator_agrees",
                    "fields": {"case_name": {"label": "agrees", "evidence": [0]}},
                    "evidence": [
                        {
                            "shows": "case_at_locator",
                            "source": {"kind": "cluster", "id": "c1"},
                            "span": {
                                "text": "sources/courtlistener/citation-lookup/1/u-s/2.json",
                                "path": "[0].clusters[0].case_name",
                            },
                        }
                    ],
                }
            },
        },
        {
            "unit": "citation",
            "id": "d1-o2",
            "is_root": True,
            "kind": "DocketCitation",
            "locator": {"start": 9, "end": 15},
            "validation": {
                "identity": {
                    "label": "WRONG_IDENTITY",
                    "source": "court_ruling",
                    "basis": "named_by_court",
                    "fields": {"date": {"label": "disagrees", "evidence": [0]}},
                    "evidence": [
                        {
                            "shows": "independent_record",
                            "source": {"kind": "docket", "id": "4271005"},
                            "span": {
                                "text": "sources/courtlistener/dockets/4271005.json",
                                "path": "date_filed",
                            },
                        }
                    ],
                }
            },
        },
        {
            "unit": "citation",
            "id": "d1-o3",
            "is_root": True,
            "kind": "FullCaseCitation",
            "identifier": {"kind": "reporter", "volume": "2", "reporter": "U.S.", "page": "3"},
            "locator": {"start": 20, "end": 25},
            "validation": {
                "identity": {
                    "label": "CORRECT_IDENTITY",
                    "basis": "record_at_locator_agrees",
                    "fields": {},
                    "evidence": [
                        {
                            "shows": "case_at_locator",
                            "source": {"kind": "opinion", "id": "op-1"},
                            "span": {"text": "sources/courtlistener/opinions/op-1.txt", "path": "header"},
                        }
                    ],
                }
            },
        },
    ]
    (annotations / "doc.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    expectations = build_expectations(annotations)
    assert len(expectations[0]["expected_retrievals"]) == 1
    assert expectations[0]["expected_retrievals"][0]["record_id"] == "c1"
    assert expectations[0]["expected_identity_decision_stage"] == "full_reporter_locator_identity_resolution"
    assert expectations[1]["expected_retrievals"][0]["record_id"] == "4271005"
    assert expectations[1]["expected_identity_decision_stage"] is None
    assert expectations[2]["expected_retrievals"] == []
    assert "Opinion/ruling" in expectations[2]["unknown_reason"]
    assert expectations[0]["annotation_provenance"] == rows[0]["validation"]["identity"]


def test_retrieval_and_first_identity_decision_are_scored_separately(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "doc.json").write_text(
        json.dumps(
            {
                "source_metadata": {"path": "doc.txt"},
                "citations": [
                    {
                        "root_id": "root",
                        "citation_id": "root",
                        "fields": {"kind": "FullCaseCitation", "locator_span": {"start": 1, "end": 5}},
                        "trace": [
                            {
                                "stage": "full_reporter_locator_exact_lookup",
                                "outcome": "found",
                                "details": {
                                    "validation": {
                                        "cluster": {"cluster_id": "c1"},
                                        "message": "untrusted prompt echo c2",
                                    }
                                },
                            },
                            {
                                "stage": "full_reporter_locator_unique_identity",
                                "outcome": "ambiguous",
                                "node_id": "root:candidate_list",
                            },
                            {
                                "stage": "full_reporter_locator_ambiguity_resolution",
                                "outcome": "resolved",
                                "node_id": "root:locator_identity_resolution",
                            },
                            {
                                "stage": "root_body_corroboration_resolution",
                                "outcome": "no_match",
                                "node_id": "root:identity_resolution",
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    expectations = [
        {
            "document": "doc",
            "kind": "FullCaseCitation",
            "locator_span": {"start": 1, "end": 5},
            "annotation_id": "doc-o1",
            "identity_label": "CORRECT_IDENTITY",
            "expected_retrievals": [
                {
                    "provider": "courtlistener",
                    "record_kind": "cluster",
                    "record_id": "c1",
                    "stage": "full_reporter_locator_exact_lookup",
                }
            ],
            "expected_identity_decision_stage": "full_reporter_locator_identity_resolution",
        }
    ]
    report = evaluate_routes(expectations, artifacts)
    row = report["roots"][0]
    assert row["all_expected_targets_retrieved"] is True
    assert row["first_identity_decision"]["stage"] == "full_reporter_locator_ambiguity_resolution"
    assert row["expected_stage_gold_judgment_correct"] is True
    assert row["target_and_decision_success"] is True


def test_stage_call_and_prompt_echo_do_not_count_as_target_retrieval(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "doc.json").write_text(
        json.dumps(
            {
                "source_metadata": {"path": "doc.txt"},
                "citations": [
                    {
                        "root_id": "root",
                        "citation_id": "root",
                        "fields": {"kind": "FullCaseCitation", "locator_span": {"start": 1, "end": 5}},
                        "trace": [
                            {
                                "stage": "full_reporter_locator_exact_lookup",
                                "outcome": "not_found",
                                "details": {"validation": {"candidate_count": 0, "message": "cluster_id c1"}},
                            },
                            {
                                "stage": "full_reporter_locator_unique_identity",
                                "outcome": "no_match",
                                "node_id": "root:locator_identity_resolution",
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    expectations = [
        {
            "document": "doc",
            "kind": "FullCaseCitation",
            "locator_span": {"start": 1, "end": 5},
            "annotation_id": "doc-o1",
            "identity_label": "WRONG_IDENTITY",
            "expected_retrievals": [
                {
                    "provider": "courtlistener",
                    "record_kind": "cluster",
                    "record_id": "c1",
                    "stage": "full_reporter_locator_exact_lookup",
                }
            ],
            "expected_identity_decision_stage": "full_reporter_locator_identity_resolution",
        }
    ]
    row = evaluate_routes(expectations, artifacts)["roots"][0]
    assert row["all_expected_targets_retrieved"] is False
    assert row["expected_stage_gold_judgment_correct"] is True
    assert row["target_and_decision_success"] is False


def test_docket_target_requires_raw_candidate_from_expected_provider(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "doc.json").write_text(
        json.dumps(
            {
                "source_metadata": {"path": "doc.txt"},
                "citations": [
                    {
                        "root_id": "root",
                        "citation_id": "root",
                        "fields": {"kind": "DocketCitation", "locator_span": {"start": 1, "end": 5}},
                        "trace": [
                            {
                                "stage": "docket_root_identity",
                                "node_id": "root:docket_root_identity:query:1:courtlistener",
                                "details": {
                                    "validation": {"attempts": [{"candidates": [{"packageId": "pkg1"}]}]}
                                },
                            },
                            {
                                "stage": "docket_root_identity",
                                "node_id": "root:docket_root_identity:query:2:govinfo:shortlist",
                                "details": {"validation": {"candidates": [{"packageId": "pkg1"}]}},
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    expected = [
        {
            "document": "doc",
            "kind": "DocketCitation",
            "locator_span": {"start": 1, "end": 5},
            "annotation_id": "doc-o1",
            "identity_label": "CORRECT_IDENTITY",
            "expected_retrievals": [
                {
                    "provider": "govinfo",
                    "record_kind": "package",
                    "record_id": "pkg1",
                    "stage": "docket_root_identity",
                }
            ],
            "expected_identity_decision_stage": None,
        }
    ]
    row = evaluate_routes(expected, artifacts)["roots"][0]
    assert row["all_expected_targets_retrieved"] is False
    payload = json.loads((artifacts / "doc.json").read_text(encoding="utf-8"))
    payload["citations"][0]["trace"].append(
        {
            "stage": "docket_root_identity",
            "node_id": "root:docket_root_identity:query:3:govinfo",
            "details": {"validation": {"attempts": [{"candidates": [{"packageId": "pkg1"}]}]}},
        }
    )
    (artifacts / "doc.json").write_text(json.dumps(payload), encoding="utf-8")
    row = evaluate_routes(expected, artifacts)["roots"][0]
    assert row["all_expected_targets_retrieved"] is True
