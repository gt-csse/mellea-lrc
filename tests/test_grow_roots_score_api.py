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
from mellea_lrc.model import FullReporterCitation, Span

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


def _quoted(text: str, quote: str, value: object) -> dict[str, object]:
    return {
        "source": {"kind": "quoted", **_span(text, quote)},
        "normalization": {"kind": "value", "value": value},
    }


def _not_stated() -> dict[str, object]:
    return {"source": {"kind": "not_stated"}, "normalization": {"kind": "unavailable"}}


def _not_applicable() -> dict[str, object]:
    return {
        "source": {"kind": "not_applicable"},
        "normalization": {"kind": "not_applicable"},
    }


def _inferred_court() -> dict[str, object]:
    return {
        "source": {"kind": "inferred", "basis": "reporter"},
        "normalization": {
            "kind": "value",
            "value": {"id": "scotus", "name": "Supreme Court of the United States"},
        },
    }


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
        "locator": _quoted(
            SOURCE,
            "550 U.S. 544",
            {"kind": "reporter", "volume": "550", "reporter": "U.S.", "page": "544"},
        ),
        "case_name": _quoted(
            SOURCE,
            "Alpha v. Beta",
            {
                "kind": "adversarial",
                "plaintiff": "Alpha",
                "defendant": "Beta",
                "subject": None,
                "partial": None,
            },
        ),
        "court": _inferred_court(),
        "date": _quoted(SOURCE, "2007", {"normalized": "2007", "precision": "year"}),
        "pin_cite": _quoted(SOURCE, "545", [{"first": 545, "last": 545, "kind": "page"}]),
        "docket_entry": _not_applicable(),
    }
    docket = {
        "unit": "citation",
        "id": "example-o02",
        "is_root": True,
        "root_id": "example-o02",
        "kind": "DocketCitation",
        "locator": _quoted(
            SOURCE,
            "No. 1:24-cv-08705",
            {"kind": "docket", "docket_number": "1:24-cv-08705"},
        ),
        "docket_entry": _quoted(SOURCE, "Dkt. 17", "17"),
        "case_name": _quoted(
            SOURCE,
            "Gamma v. Delta",
            {
                "kind": "adversarial",
                "plaintiff": "Gamma",
                "defendant": "Delta",
                "subject": None,
                "partial": None,
            },
        ),
        "court": _quoted(
            SOURCE,
            "S.D.N.Y.",
            {"id": "nysd", "name": "District Court, S.D. New York"},
        ),
        "date": _quoted(SOURCE, "2024", {"normalized": "2024", "precision": "year"}),
        "pin_cite": _not_stated(),
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


def _write_single_reporter_source(tmp_path: Path, text: str, case_name: dict[str, object]) -> Path:
    set_dir = tmp_path / "primary"
    source_dir = set_dir / "documents_txt"
    annotation_dir = set_dir / "documents"
    source_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    source_path = source_dir / "example.txt"
    source_path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    header = {
        "unit": "header",
        "dataset": "primary",
        "document": source_path.name,
        "text": {
            "path": "primary/documents_txt/example.txt",
            "sha256": digest,
            "length": len(text),
        },
    }
    root = {
        "unit": "citation",
        "id": "example-o01",
        "is_root": True,
        "root_id": "example-o01",
        "kind": "FullCaseCitation",
        "locator": _quoted(
            text,
            "550 U.S. 544",
            {"kind": "reporter", "volume": "550", "reporter": "U.S.", "page": "544"},
        ),
        "case_name": case_name,
        "court": _inferred_court(),
        "date": _not_stated(),
        "pin_cite": _not_stated(),
        "docket_entry": _not_applicable(),
    }
    (annotation_dir / "example.jsonl").write_text(
        "\n".join(json.dumps(row) for row in (header, root)) + "\n", encoding="utf-8"
    )
    (set_dir / "documents.json").write_text(
        json.dumps({"documents": {source_path.name: {"sha256": digest, "length": len(text)}}}),
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


def test_docket_root_review_score_handles_mixed_citation_types(
    annotated_document: Document,
) -> None:
    document = annotated_document.complete("11_docket_root_llm_reassignment")
    stage = evaluation.score_docket_root_llm_reassignment(document)
    assert stage.metrics["root_assignment"] == evaluation.Precision(0, 0)
    assert evaluation.score_grow_roots(document).stages[-1] == stage


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
        "span_overlap": evaluation.RecallScore(2, 2),
        "normalization": evaluation.FieldScore(2, 2, 2),
    }
    assert score.root_fields["court"] == {
        "span": evaluation.FieldScore(2, 2, 2),
        "span_overlap": evaluation.RecallScore(2, 2),
        "normalization": evaluation.FieldScore(2, 2, 2),
    }
    for name in ("full_reporter_locator", "docket_locator", "docket_entry"):
        assert score.root_fields[name] == {
            "span": evaluation.FieldScore(1, 1, 1),
            "span_overlap": evaluation.RecallScore(1, 1),
            "normalization": evaluation.FieldScore(1, 1, 1),
        }
    for name in ("date", "pin_cite"):
        assert score.root_fields[name] == {
            "span": evaluation.FieldScore(2, 2, 2),
            "span_overlap": evaluation.RecallScore(2, 2),
            "normalization": evaluation.FieldScore(2, 2, 2),
        }
    assert score.root_fields["overall_locator"] == {
        "span": evaluation.FieldScore(2, 2, 2),
        "span_overlap": evaluation.RecallScore(2, 2),
        "normalization": evaluation.FieldScore(2, 2, 2),
    }
    assert tuple(score.root_fields)[:3] == (
        "full_reporter_locator",
        "docket_locator",
        "overall_locator",
    )


def _validated_document_with_changed_fields(document: Document) -> Document:
    for stage in (
        "11_docket_root_llm_reassignment",
        "12.1_reporter_root_lookup_cluster_retrieval",
        "13.1_reporter_root_lookup_unique_rule_judgment",
        "12.2_reporter_root_lookup_docket_retrieval",
        "13.2_reporter_root_lookup_ambiguous_rule_judgment",
        "14_reporter_root_lookup_unique_llm_judgment",
        "15_reporter_root_lookup_ambiguous_llm_judgment",
        "16_docket_root_lookup_courtlistener_retrieval",
        "17_docket_root_lookup_courtlistener_llm_review",
        "18_docket_root_lookup_govinfo_retrieval",
    ):
        document = document.complete(stage)
    reporter = next(root for root in document.roots if isinstance(root, FullReporterCitation))
    changed = reporter.record("19_docket_root_lookup_govinfo_llm_review")
    for method, quote in (
        ("with_case_name", "Gamma v. Delta"),
        ("with_court", "S.D.N.Y."),
        ("with_date", "2024"),
    ):
        source_span = _span(SOURCE, quote)
        changed = getattr(changed, method)(SOURCE, Span(source_span["start"], source_span["end"]))
    return document.replace_citation(changed).complete("19_docket_root_lookup_govinfo_llm_review")


def test_validation_checkpoint_scores_latest_root_fields_without_changing_baseline(
    annotated_document: Document,
) -> None:
    baseline_document = annotated_document.complete("11_docket_root_llm_reassignment")
    baseline = evaluation.score_grow_roots(baseline_document)
    assert baseline.validated_root_fields is None

    validated = _validated_document_with_changed_fields(annotated_document)
    score = evaluation.score_grow_roots(validated)
    assert score.root_fields == baseline.root_fields
    assert score.stages == baseline.stages
    assert score.validated_root_fields is not None
    assert set(score.validated_root_fields) == {"case_name", "court", "date"}
    for field in ("case_name", "court", "date"):
        assert score.root_fields[field] == {
            "span": evaluation.FieldScore(2, 2, 2),
            "span_overlap": evaluation.RecallScore(2, 2),
            "normalization": evaluation.FieldScore(2, 2, 2),
        }
        assert score.validated_root_fields[field] == {
            "span": evaluation.FieldScore(1, 2, 2),
            "span_overlap": evaluation.RecallScore(1, 2),
            "normalization": evaluation.FieldScore(1, 2, 2),
        }

    assert score.as_dict()["validated_root_fields"]["case_name"]["span"] == {
        "correct": 1,
        "predicted": 2,
        "gold": 2,
        "precision": 0.5,
        "recall": 0.5,
    }
    report = evaluation.render_grow_roots(score, include_stages=False)
    assert report.count("| case_name |") == 2
    assert "| case_name | 1/2 (50.0%)" in report
    assert "19_docket_root_lookup_govinfo_llm_review" in report

    doubled = score + score
    assert doubled.validated_root_fields is not None
    assert doubled.validated_root_fields["case_name"]["span"] == evaluation.FieldScore(2, 4, 4)
    assert doubled.validated_root_fields["case_name"]["span_overlap"] == evaluation.RecallScore(2, 4)


def test_later_body_reading_does_not_change_validation_checkpoint_score(
    annotated_document: Document,
) -> None:
    validated = _validated_document_with_changed_fields(annotated_document)
    expected = evaluation.score_grow_roots(validated)
    for stage in (
        "20_locator_body_courtlistener_opinion_retrieval",
        "21_locator_body_courtlistener_recap_retrieval",
        "22_locator_body_govinfo_opinion_retrieval",
    ):
        validated = validated.complete(stage)
    reporter = next(root for root in validated.roots if isinstance(root, FullReporterCitation))
    changed = reporter.record("23_locator_body_llm_judgment")
    for method, quote in (("with_case_name", "Alpha v. Beta"), ("with_date", "2007")):
        source_span = _span(SOURCE, quote)
        changed = getattr(changed, method)(SOURCE, Span(source_span["start"], source_span["end"]))
    reviewed = validated.replace_citation(changed).complete("23_locator_body_llm_judgment")
    assert (
        reviewed.roots[0].case_name[-1]
        != reviewed.get_stage("19_docket_root_lookup_govinfo_llm_review").roots[0].case_name[-1]
    )
    assert evaluation.score_grow_roots(reviewed) == expected


def test_overall_locator_subtotal_adds_counts_across_documents(
    annotated_document: Document,
) -> None:
    score = evaluation.score_grow_roots(annotated_document)
    combined = score + score
    assert combined.root_fields["overall_locator"] == {
        measure: combined.root_fields["full_reporter_locator"][measure]
        + combined.root_fields["docket_locator"][measure]
        for measure in ("span", "span_overlap", "normalization")
    }
    assert combined.root_fields["overall_locator"]["span"] == evaluation.FieldScore(4, 4, 4)


def test_normalization_disagreement_does_not_change_span_score(
    annotated_document: Document,
) -> None:
    source = Path(annotated_document.source_path)
    annotation = source.parent.parent / "documents" / f"{source.stem}.jsonl"
    rows = [json.loads(line) for line in annotation.read_text(encoding="utf-8").splitlines()]
    rows[1]["case_name"]["normalization"]["value"]["plaintiff"] = "Different party"
    annotation.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    stage = evaluation.score_case_names(annotated_document)
    workflow = evaluation.score_grow_roots(annotated_document)
    assert stage.metrics["span"] == evaluation.Precision(2, 2)
    assert stage.metrics["normalization"] == evaluation.Precision(1, 2)
    assert workflow.root_fields["case_name"]["span"] == evaluation.FieldScore(2, 2, 2)
    assert workflow.root_fields["case_name"]["normalization"] == evaluation.FieldScore(1, 2, 2)


def test_partial_nonroot_identifier_is_not_scored_as_normalization(
    annotated_document: Document,
) -> None:
    source = Path(annotated_document.source_path)
    annotation = source.parent.parent / "documents" / f"{source.stem}.jsonl"
    rows = [json.loads(line) for line in annotation.read_text(encoding="utf-8").splitlines()]
    rows[1]["is_root"] = False
    rows[1]["locator"] = _span(SOURCE, "550 U.S. 544")
    rows[1]["identifier"] = {"kind": "reporter", "volume": "550"}
    annotation.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    score = evaluation.score_full_reporter_locators(annotated_document)
    assert score.metrics["span"] == evaluation.Precision(1, 1)
    assert score.metrics["normalization"] == evaluation.Precision(0, 0)


def test_overlap_recall_accepts_partial_locator_and_field_spans(
    annotated_document: Document,
) -> None:
    source = Path(annotated_document.source_path)
    annotation = source.parent.parent / "documents" / f"{source.stem}.jsonl"
    rows = [json.loads(line) for line in annotation.read_text(encoding="utf-8").splitlines()]
    rows[1]["locator"]["source"] = {"kind": "quoted", **_span(SOURCE, "550 U.S. 544,")}
    rows[1]["case_name"]["source"] = {"kind": "quoted", **_span(SOURCE, "Alpha v. Beta,")}
    annotation.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    score = evaluation.score_grow_roots(annotated_document)
    assert score.root_fields["full_reporter_locator"]["span"] == evaluation.FieldScore(0, 1, 1)
    assert score.root_fields["full_reporter_locator"]["span_overlap"] == evaluation.RecallScore(1, 1)
    assert score.root_fields["case_name"]["span"] == evaluation.FieldScore(1, 2, 2)
    assert score.root_fields["case_name"]["span_overlap"] == evaluation.RecallScore(2, 2)


def test_overlap_recall_cannot_credit_one_predicted_root_twice(tmp_path: Path) -> None:
    text = "550 U.S. 544, as cited."
    source = _write_single_reporter_source(tmp_path, text, _not_stated())
    document = asyncio.run(grow_roots(Document.from_source(source)))
    annotation = source.parent.parent / "documents" / "example.jsonl"
    rows = [json.loads(line) for line in annotation.read_text(encoding="utf-8").splitlines()]
    second = json.loads(json.dumps(rows[1]))
    second["id"] = second["root_id"] = "example-o02"
    second["locator"]["source"] = {"kind": "quoted", **_span(text, "550 U.S. 544,")}
    annotation.write_text("\n".join(json.dumps(row) for row in (*rows, second)) + "\n", encoding="utf-8")

    score = evaluation.score_grow_roots(document)
    assert score.root_fields["full_reporter_locator"]["span_overlap"] == evaluation.RecallScore(1, 2)
    assert score.root_fields["case_name"]["span_overlap"] == evaluation.RecallScore(1, 2)


def test_workflow_rejects_a_missing_mandatory_checkpoint(annotated_document: Document) -> None:
    incomplete = annotated_document.get_stage("2_docket_locators").complete("10_roots")
    with pytest.raises(ValueError, match="Incomplete grow_roots workflow"):
        evaluation.score_grow_roots(incomplete)


@pytest.mark.parametrize("malformation", ["missing", "null", "legacy"])
def test_malformed_root_gold_is_rejected(annotated_document: Document, malformation: str) -> None:
    source = Path(annotated_document.source_path)
    annotation = source.parent.parent / "documents" / f"{source.stem}.jsonl"
    rows = [json.loads(line) for line in annotation.read_text(encoding="utf-8").splitlines()]
    if malformation == "missing":
        rows[1].pop("court")
    elif malformation == "null":
        rows[1]["court"] = None
    else:
        rows[1]["court"] = {"id": "scotus", "name": "Supreme Court of the United States"}
    annotation.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        evaluation.score_courts(annotated_document)
    with pytest.raises(ValueError):
        evaluation.score_grow_roots(annotated_document)


def test_absent_case_name_scores_as_correct_only_for_not_stated(tmp_path: Path) -> None:
    source = _write_single_reporter_source(tmp_path, "550 U.S. 544.", _not_stated())
    document = asyncio.run(grow_roots(Document.from_source(source)))
    score = evaluation.score_grow_roots(document)
    assert score.root_fields["case_name"] == {
        "span": evaluation.FieldScore(1, 1, 1),
        "span_overlap": evaluation.RecallScore(1, 1),
        "normalization": evaluation.FieldScore(1, 1, 1),
    }
    assert score.root_fields["date"]["normalization"] == evaluation.FieldScore(1, 1, 1)
    assert score.root_fields["pin_cite"]["normalization"] == evaluation.FieldScore(1, 1, 1)


def test_quoted_partial_case_name_scores_with_its_own_typed_value(tmp_path: Path) -> None:
    text = "Gucci America, 550 U.S. 544."
    partial = {
        "source": {"kind": "quoted", **_span(text, "Gucci America")},
        "normalization": {
            "kind": "value",
            "value": {
                "kind": "partial",
                "plaintiff": None,
                "defendant": None,
                "subject": None,
                "partial": "Gucci America",
            },
        },
    }
    source = _write_single_reporter_source(tmp_path, text, partial)
    document = asyncio.run(grow_roots(Document.from_source(source)))

    assert evaluation.score_case_names(document).metrics == {
        "span": evaluation.Precision(1, 1),
        "normalization": evaluation.Precision(1, 1),
    }
    assert evaluation.score_grow_roots(document).root_fields["case_name"] == {
        "span": evaluation.FieldScore(1, 1, 1),
        "span_overlap": evaluation.RecallScore(1, 1),
        "normalization": evaluation.FieldScore(1, 1, 1),
    }


def test_not_stated_gold_rejects_a_predicted_case_name(tmp_path: Path) -> None:
    source = _write_single_reporter_source(tmp_path, "Alpha v. Beta, 550 U.S. 544.", _not_stated())
    document = asyncio.run(grow_roots(Document.from_source(source)))
    score = evaluation.score_grow_roots(document)
    assert score.root_fields["case_name"] == {
        "span": evaluation.FieldScore(0, 1, 1),
        "span_overlap": evaluation.RecallScore(0, 1),
        "normalization": evaluation.FieldScore(0, 1, 1),
    }


def test_failed_normalization_matches_only_quoted_unavailable_gold(tmp_path: Path) -> None:
    text = "Smith v. ?, 550 U.S. 544."
    unavailable = {
        "source": {"kind": "quoted", **_span(text, "Smith v. ?")},
        "normalization": {"kind": "unavailable", "reason": "underdetermined"},
    }
    source = _write_single_reporter_source(tmp_path, text, unavailable)
    document = asyncio.run(grow_roots(Document.from_source(source)))
    reading = document.roots[0].case_name[-1]
    assert reading.normalizable is False

    score = evaluation.score_grow_roots(document)
    assert score.root_fields["case_name"] == {
        "span": evaluation.FieldScore(1, 1, 1),
        "span_overlap": evaluation.RecallScore(1, 1),
        "normalization": evaluation.FieldScore(1, 1, 1),
    }
    assert evaluation.score_case_names(document).metrics["normalization"] == evaluation.Precision(1, 1)

    annotation = source.parent.parent / "documents" / "example.jsonl"
    rows = [json.loads(line) for line in annotation.read_text(encoding="utf-8").splitlines()]
    rows[1]["case_name"]["source"] = {"kind": "quoted", **_span(text, "Smith")}
    annotation.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    wrong_quote = evaluation.score_grow_roots(document)
    assert wrong_quote.root_fields["case_name"]["span"] == evaluation.FieldScore(0, 1, 1)
    assert wrong_quote.root_fields["case_name"]["span_overlap"] == evaluation.RecallScore(1, 1)
    assert wrong_quote.root_fields["case_name"]["normalization"] == evaluation.FieldScore(0, 1, 1)

    rows[1]["case_name"]["source"] = {"kind": "quoted", **_span(text, "Smith v. ?")}
    rows[1]["case_name"]["normalization"] = {
        "kind": "value",
        "value": {
            "kind": "adversarial",
            "plaintiff": "Smith",
            "defendant": "Jones",
            "subject": None,
            "partial": None,
        },
    }
    annotation.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    changed = evaluation.score_grow_roots(document)
    assert changed.root_fields["case_name"]["span"] == evaluation.FieldScore(1, 1, 1)
    assert changed.root_fields["case_name"]["normalization"] == evaluation.FieldScore(0, 1, 1)


def test_normalizable_case_name_does_not_match_unavailable_gold(tmp_path: Path) -> None:
    text = "Alpha v. Beta, 550 U.S. 544."
    unavailable = {
        "source": {"kind": "quoted", **_span(text, "Alpha v. Beta")},
        "normalization": {"kind": "unavailable", "reason": "underdetermined"},
    }
    source = _write_single_reporter_source(tmp_path, text, unavailable)
    document = asyncio.run(grow_roots(Document.from_source(source)))
    assert document.roots[0].case_name[-1].normalizable is True
    score = evaluation.score_grow_roots(document)
    assert score.root_fields["case_name"]["span"] == evaluation.FieldScore(1, 1, 1)
    assert score.root_fields["case_name"]["span_overlap"] == evaluation.RecallScore(1, 1)
    assert score.root_fields["case_name"]["normalization"] == evaluation.FieldScore(0, 1, 1)


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
        "| **overall_locator subtotal** | 2/2 (100.0%) | 2/2 (100.0%) | 2/2 (100.0%) | 2/2 (100.0%) | 2/2 (100.0%) |"
    ) in report

    summary_only = evaluation.render_grow_roots(score, include_stages=False)
    assert "## Root fields\n" in summary_only
    assert all(f"## {stage}\n" not in summary_only for stage in STAGE_SCORERS)

    with pytest.raises(ValueError, match="missing or out of order"):
        evaluation.render_grow_roots(evaluation.WorkflowScore(score.stages[:-1], score.root_fields))
