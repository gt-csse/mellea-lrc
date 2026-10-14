"""Reports preserve atomic scores and distinguish completed stage boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluations import validate_all_citation_content
from evaluations import score_run as scoring
from evaluations.grow_leaves import WorkflowScore as LeafWorkflowScore
from evaluations.score_run import score_run
from evaluations.score_types import (
    FieldScore,
    Precision,
    SubstageScore,
    group_substage_records,
    render_stage_sections,
    substage_heading,
)
from mellea_lrc.model import Document
from mellea_lrc.model.execution import get_workflow


def _prepared_opinions() -> Document:
    document = Document.from_source("A filing without citations.")
    for substage in get_workflow("validate_all_citation_content").stages[0].substages:
        document = document.complete_substage(substage.name)
    return document


def test_group_completion_uses_the_durable_marker_and_preserves_atomic_metrics() -> None:
    atomic = _prepared_opinions()
    grouped = atomic.complete_stage("validate_all_citation_content.prepare_root_opinions")
    partial = validate_all_citation_content.score_validate_all_citation_content(atomic)
    completed = validate_all_citation_content.score_validate_all_citation_content(grouped)

    assert partial.substages == completed.substages
    assert partial.as_dict()["stages"][0]["completed"] is False
    assert completed.as_dict()["stages"][0]["completed"] is True
    assert partial.as_dict()["stages"][0]["substages"] == [
        substage.as_dict() for substage in completed.substages
    ]
    assert (
        "## 1 validate_all_citation_content.prepare_root_opinions (incomplete)"
        in validate_all_citation_content.render_validate_all_citation_content(partial)
    )
    assert (
        "## 1 validate_all_citation_content.prepare_root_opinions\n"
        in validate_all_citation_content.render_validate_all_citation_content(completed)
    )
    assert grouped.get_substage(validate_all_citation_content.PAGE_INDEX_SUBSTAGE) == atomic
    assert validate_all_citation_content.score_retrieve_reporter_root_opinions(
        grouped
    ) == validate_all_citation_content.score_retrieve_reporter_root_opinions(atomic)


def test_aggregate_group_is_completed_only_for_all_contributing_documents() -> None:
    atomic = _prepared_opinions()
    grouped = atomic.complete_stage("validate_all_citation_content.prepare_root_opinions")
    partial = validate_all_citation_content.score_validate_all_citation_content(atomic)
    completed = validate_all_citation_content.score_validate_all_citation_content(grouped)

    assert (completed + completed).as_dict()["stages"][0]["completed"] is True
    mixed = completed + partial
    assert mixed.as_dict()["stages"][0]["completed"] is False
    assert mixed.substages == (completed + completed).substages


def test_optional_substages_need_no_placeholder_and_summaries_keep_their_denominators() -> None:
    stage = get_workflow("grow_leaves").stages[3]
    substages = tuple(
        SubstageScore(name, {"attribution": Precision(2, 3)}) for name in stage.required_substages
    )
    score = LeafWorkflowScore(
        substages,
        {"all_leaves": FieldScore(2, 3, 5)},
        {"all_leaves": FieldScore(1, 2, 5)},
        (stage.name,),
    )
    result = score.as_dict()
    assert result["stages"] == [
        {
            "stage": stage.name,
            "completed": True,
            "substages": [substage.as_dict() for substage in substages],
        }
    ]
    assert result["leaf_spans"]["all_leaves"] == FieldScore(2, 3, 5).as_dict()
    assert result["leaf_attribution"]["all_leaves"] == FieldScore(1, 2, 5).as_dict()
    assert stage.substages[-1].optional is True
    assert stage.substages[-1].name not in {item["substage"] for item in result["stages"][0]["substages"]}


def test_report_order_and_local_numbering_come_from_the_catalog() -> None:
    records = [
        {
            "substage": "validate_roots_citation_identity.lookup_docket_roots.retrieve_govinfo_dockets",
            "record_coverage": {"rate": None},
        },
        {
            "substage": "validate_roots_citation_identity.lookup_reporter_roots.retrieve_clusters",
            "record_coverage": {"rate": 0.5},
        },
        {
            "substage": "validate_roots_citation_identity.lookup_reporter_roots.review_ambiguous_llm",
            "metrics": {},
        },
    ]
    grouped = group_substage_records("validate_roots_citation_identity", records)
    flattened = [record for stage in grouped for record in stage["substages"]]
    assert flattened == [records[1], records[2], records[0]]
    assert all(stage["completed"] is False for stage in grouped)
    rendered = render_stage_sections(
        "validate_roots_citation_identity",
        ((record["substage"], substage_heading(record["substage"])) for record in records),
    )
    assert rendered == [
        "## 1 validate_roots_citation_identity.lookup_reporter_roots (incomplete)",
        "### 1.1 validate_roots_citation_identity.lookup_reporter_roots.retrieve_clusters",
        "### 1.6 validate_roots_citation_identity.lookup_reporter_roots.review_ambiguous_llm",
        "## 2 validate_roots_citation_identity.lookup_docket_roots (incomplete)",
        "### 2.3 validate_roots_citation_identity.lookup_docket_roots.retrieve_govinfo_dockets",
    ]


def test_duplicate_or_foreign_substage_data_cannot_be_hidden_by_grouping() -> None:
    record = {"substage": "grow_leaves.grow_id_leaves.find_id_citations", "metrics": {}}
    with pytest.raises(ValueError, match="Duplicate substage score"):
        group_substage_records("grow_leaves", [record, record])
    with pytest.raises(ValueError, match="Unknown grow_roots substages"):
        group_substage_records("grow_roots", [record])


@pytest.mark.parametrize("workflows", [None, ("grow_roots",), ("validate_all_citation_content",)])
def test_empty_saved_run_creates_no_workflow_reports(
    tmp_path: Path, workflows: tuple[str, ...] | None
) -> None:
    (tmp_path / "documents").mkdir()
    (tmp_path / "run.json").write_text(json.dumps({"status": "complete", "set": "primary", "filings": []}))

    with pytest.raises(ValueError, match="Cannot score a run without Documents"):
        score_run(tmp_path, workflows)
    assert {path.name for path in tmp_path.iterdir()} == {"documents", "run.json"}


@pytest.mark.parametrize(
    "completed",
    [
        ("grow_roots",),
        ("grow_roots", "grow_leaves"),
        ("grow_roots", "validate_roots_citation_identity"),
        ("grow_roots", "validate_roots_citation_identity", "grow_leaves", "validate_all_citation_content"),
    ],
)
def test_default_reports_only_score_workflows_present_in_the_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, completed: tuple[str, ...]
) -> None:
    document = Document.from_source("A filing without citations.")
    for name in completed:
        document = document.complete_substage(get_workflow(name).stages[0].substages[0].name)
    (tmp_path / "documents").mkdir()
    (tmp_path / "documents" / "example.txt.json").write_text(document.model_dump_json())
    (tmp_path / "run.json").write_text(
        json.dumps({"status": "complete", "set": "fixture", "filings": ["example.txt"]})
    )
    scored = []

    def scorer(name):
        def score(value):
            assert get_workflow(name).stages[0].substages[0].name in value.substage_runs
            scored.append(name)
            return Precision(0, 0)

        return score

    monkeypatch.setattr(
        scoring,
        "_WORKFLOWS",
        {name: (scorer(name), lambda score, *, set_name: set_name) for name in scoring._WORKFLOWS},
    )

    reports = score_run(tmp_path)

    assert tuple(reports) == completed
    assert tuple(scored) == completed
    assert {path.stem for path in tmp_path.glob("*.md")} == set(completed)
