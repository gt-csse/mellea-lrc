"""The public grow-roots scorers take a saved Document and honor stage boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from evaluations import grow_roots as evaluation
from mellea_lrc.api import Document, grow_roots

SOURCE = (
    "Alpha v. Beta, 550 U.S. 544, 545 (2007). Gamma v. Delta, No. 1:24-cv-08705, Dkt. 17 (S.D.N.Y. 2024)."
)

STAGE_SCORERS = {
    "1_full_reporter_locators": "score_full_reporter_locators",
    "2_docket_locators": "score_docket_locators",
    "3_docket_locator_site_hunting": "score_docket_locator_site_hunting",
    "4_docket_entries": "score_docket_entries",
    "5_colocations": "score_colocations",
    "6_case_names": "score_case_names",
    "7_courts": "score_courts",
    "8_dates": "score_dates",
    "9_pin_cites": "score_pin_cites",
    "10_roots": "score_roots",
}
STAGE_RENDERERS = {stage: name.replace("score_", "render_") for stage, name in STAGE_SCORERS.items()}


def _span(text: str, quote: str) -> dict[str, str | int]:
    start = text.index(quote)
    return {"start": start, "end": start + len(quote), "quote": quote}


def _write_annotated_source(tmp_path: Path) -> Path:
    set_dir = tmp_path / "primary"
    source_dir = set_dir / "documents_txt"
    annotation_dir = set_dir / "documents"
    source_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    source_path = source_dir / "example.txt"
    source_path.write_text(SOURCE, encoding="utf-8")
    digest = hashlib.sha256(SOURCE.encode("utf-8")).hexdigest()

    header = {
        "unit": "header",
        "dataset": "primary",
        "document": source_path.name,
        "text": {
            "path": "primary/documents_txt/example.txt",
            "sha256": digest,
            "length": len(SOURCE),
        },
    }
    reporter = {
        "unit": "citation",
        "id": "example-o01",
        "is_root": True,
        "root_id": "example-o01",
        "kind": "FullCaseCitation",
        "identifier": {"kind": "reporter", "volume": "550", "reporter": "U.S.", "page": "544"},
        "locator": _span(SOURCE, "550 U.S. 544"),
        "case_name": {
            **_span(SOURCE, "Alpha v. Beta"),
            "normalized": {
                "kind": "adversarial",
                "plaintiff": "Alpha",
                "defendant": "Beta",
                "subject": None,
            },
        },
        "court": {"id": "scotus", "name": "Supreme Court of the United States", "how": "reporter"},
        "date": {**_span(SOURCE, "2007"), "normalized": "2007", "precision": "year"},
        "pin_cite": {
            **_span(SOURCE, "545"),
            "normalized": [{"first": 545, "last": 545, "kind": "page"}],
        },
    }
    docket = {
        "unit": "citation",
        "id": "example-o02",
        "is_root": True,
        "root_id": "example-o02",
        "kind": "DocketCitation",
        "identifier": {"kind": "docket", "docket_number": "1:24-cv-08705"},
        "locator": _span(SOURCE, "No. 1:24-cv-08705"),
        "docket_entry": {**_span(SOURCE, "Dkt. 17"), "number": "17"},
        "case_name": {
            **_span(SOURCE, "Gamma v. Delta"),
            "normalized": {
                "kind": "adversarial",
                "plaintiff": "Gamma",
                "defendant": "Delta",
                "subject": None,
            },
        },
        "court": {
            **_span(SOURCE, "S.D.N.Y."),
            "id": "nysd",
            "name": "District Court, S.D. New York",
            "how": "stated",
        },
        "date": {**_span(SOURCE, "2024"), "normalized": "2024", "precision": "year"},
    }
    annotation_path = annotation_dir / "example.jsonl"
    annotation_path.write_text(
        "\n".join(json.dumps(row) for row in (header, reporter, docket)) + "\n",
        encoding="utf-8",
    )
    (set_dir / "documents.json").write_text(
        json.dumps({"documents": {source_path.name: {"sha256": digest, "length": len(SOURCE)}}}),
        encoding="utf-8",
    )
    return source_path


@pytest.fixture
def annotated_document(tmp_path: Path) -> Document:
    source_path = _write_annotated_source(tmp_path)
    return asyncio.run(grow_roots(Document.from_source(source_path), hunt_dockets=True))


@pytest.mark.parametrize("name", (*STAGE_SCORERS.values(), "score_grow_roots"))
def test_public_scorer_has_one_document_parameter(name: str) -> None:
    scorer = getattr(evaluation, name)
    parameters = tuple(inspect.signature(scorer).parameters.values())
    assert len(parameters) == 1
    assert parameters[0].name == "document"
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters[0].default is inspect.Parameter.empty
    assert not hasattr(evaluation, "score_stage")
    assert not hasattr(evaluation, "_field_stage")
    assert not hasattr(evaluation, "score_grow_roots_workflow")


@pytest.mark.parametrize("stage,name", STAGE_SCORERS.items())
def test_stage_scorer_uses_its_exact_checkpoint(annotated_document: Document, stage: str, name: str) -> None:
    scorer: Callable[[Document], object] = getattr(evaluation, name)
    checkpoint = annotated_document.get_stage(stage)
    restored = Document.model_validate_json(annotated_document.model_dump_json())

    assert scorer(annotated_document) == scorer(checkpoint)
    assert scorer(restored) == scorer(checkpoint)


def test_optional_hunting_scorer_requires_completed_stage(annotated_document: Document) -> None:
    without_hunting = annotated_document.get_stage("2_docket_locators")
    with pytest.raises(KeyError, match="Stage has not run"):
        evaluation.score_docket_locator_site_hunting(without_hunting)


def test_workflow_scorer_accepts_saved_annotated_document(annotated_document: Document) -> None:
    restored = Document.model_validate_json(annotated_document.model_dump_json())
    assert evaluation.score_grow_roots(annotated_document) == evaluation.score_grow_roots(restored)


def test_workflow_reports_each_root_field_with_annotated_denominators(
    annotated_document: Document,
) -> None:
    score = evaluation.score_grow_roots(annotated_document)
    stages = {item.stage: item.metrics for item in score.stages}
    assert stages["6_case_names"] == {
        "span": evaluation.Precision(2, 2),
        "normalization": evaluation.Precision(2, 2),
    }
    assert stages["7_courts"] == {
        "span": evaluation.Precision(1, 1),
        "normalization": evaluation.Precision(2, 2),
    }
    assert score.root_fields["case_name"] == {
        "span": evaluation.FieldScore(2, 2, 2),
        "normalization": evaluation.FieldScore(2, 2, 2),
    }
    assert score.root_fields["court"] == {
        "span": evaluation.FieldScore(1, 1, 1),
        "normalization": evaluation.FieldScore(2, 2, 2),
    }
    for name in ("full_reporter_locator", "docket_locator", "docket_entry", "pin_cite"):
        assert score.root_fields[name] == {
            "span": evaluation.FieldScore(1, 1, 1),
            "normalization": evaluation.FieldScore(1, 1, 1),
        }
    assert score.root_fields["overall_locator"] == {
        "span": evaluation.FieldScore(2, 2, 2),
        "normalization": evaluation.FieldScore(2, 2, 2),
    }
    assert tuple(score.root_fields)[:3] == (
        "full_reporter_locator",
        "docket_locator",
        "overall_locator",
    )


def test_overall_locator_subtotal_adds_counts_across_documents(
    annotated_document: Document,
) -> None:
    score = evaluation.score_grow_roots(annotated_document)
    combined = score + score
    assert combined.root_fields["overall_locator"] == {
        measure: combined.root_fields["full_reporter_locator"][measure]
        + combined.root_fields["docket_locator"][measure]
        for measure in ("span", "normalization")
    }
    assert combined.root_fields["overall_locator"]["span"] == evaluation.FieldScore(4, 4, 4)


def test_normalization_disagreement_does_not_change_span_score(
    annotated_document: Document,
) -> None:
    source = Path(annotated_document.source_path)
    annotation = source.parent.parent / "documents" / f"{source.stem}.jsonl"
    rows = [json.loads(line) for line in annotation.read_text(encoding="utf-8").splitlines()]
    rows[1]["case_name"]["normalized"]["plaintiff"] = "Different party"
    annotation.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    stage = evaluation.score_case_names(annotated_document)
    workflow = evaluation.score_grow_roots(annotated_document)
    assert stage.metrics["span"] == evaluation.Precision(2, 2)
    assert stage.metrics["normalization"] == evaluation.Precision(1, 2)
    assert workflow.root_fields["case_name"]["span"] == evaluation.FieldScore(2, 2, 2)
    assert workflow.root_fields["case_name"]["normalization"] == evaluation.FieldScore(1, 2, 2)


def test_workflow_rejects_a_missing_mandatory_checkpoint(annotated_document: Document) -> None:
    incomplete = annotated_document.get_stage("2_docket_locators").complete("10_roots")
    with pytest.raises(ValueError, match="Incomplete grow_roots workflow"):
        evaluation.score_grow_roots(incomplete)


def test_unannotated_normalization_and_absent_field_are_distinct(
    annotated_document: Document,
) -> None:
    source = Path(annotated_document.source_path)
    annotation = source.parent.parent / "documents" / f"{source.stem}.jsonl"
    rows = [json.loads(line) for line in annotation.read_text(encoding="utf-8").splitlines()]
    rows[1].pop("court")
    rows[2]["case_name"].pop("normalized")
    annotation.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    courts = evaluation.score_courts(annotated_document).metrics
    names = evaluation.score_case_names(annotated_document).metrics
    assert courts["normalization"] == evaluation.Precision(1, 2)
    assert names["normalization"] == evaluation.Precision(1, 1)
    workflow = evaluation.score_grow_roots(annotated_document)
    assert workflow.root_fields["court"]["normalization"] == evaluation.FieldScore(1, 2, 1)
    assert workflow.root_fields["case_name"]["normalization"] == evaluation.FieldScore(1, 1, 1)


@pytest.mark.parametrize("stage,name", STAGE_RENDERERS.items())
def test_each_stage_has_its_own_renderer(annotated_document: Document, stage: str, name: str) -> None:
    stage_score = getattr(evaluation, STAGE_SCORERS[stage])(annotated_document)
    rendered = getattr(evaluation, name)(stage_score)
    assert rendered.startswith(f"## {stage}\n")
    assert "precision" in rendered
    other = next(item for item in STAGE_SCORERS if item != stage)
    with pytest.raises(ValueError, match="Expected"):
        getattr(evaluation, name)(getattr(evaluation, STAGE_SCORERS[other])(annotated_document))


def test_workflow_renderer_includes_numbered_stages_in_order_by_default(
    annotated_document: Document,
) -> None:
    score = evaluation.score_grow_roots(annotated_document)
    report = evaluation.render_grow_roots(score, set_name="primary")
    positions = [report.index(f"## {stage}\n") for stage in STAGE_SCORERS]
    assert positions == sorted(positions)
    assert report.index("## Root fields\n") > positions[-1]
    assert "Docket site hunting: included" in report
    assert (
        "| **overall_locator subtotal** | 2/2 (100.0%) | 2/2 (100.0%) | 2/2 (100.0%) | 2/2 (100.0%) |"
    ) in report

    summary_only = evaluation.render_grow_roots(score, include_stages=False)
    assert "## Root fields\n" in summary_only
    assert all(f"## {stage}\n" not in summary_only for stage in STAGE_SCORERS)

    with pytest.raises(ValueError, match="missing or out of order"):
        evaluation.render_grow_roots(evaluation.WorkflowScore(score.stages[:-1], score.root_fields))
