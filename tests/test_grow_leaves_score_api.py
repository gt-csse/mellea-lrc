"""Source-grounded leaf evaluation and score_run integration tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from evaluations import grow_leaves as evaluation
from evaluations.annotations import citation_annotations
from evaluations.score_run import score_run
from evaluations.score_types import FieldScore, Precision
from mellea_lrc.api import Document, grow_roots
from mellea_lrc.extraction.id_attribution import attribute_id_citations
from mellea_lrc.extraction.leaf_attribution_review.reviewer import LeafReviewOutcome
from mellea_lrc.model.citations import IdCitation, LeafReviewDecision, ReferenceCitation, SupraCitation
from mellea_lrc.model.span import Span
from mellea_lrc.workflows.grow_leaves import grow_leaves

TEXT = "Smith v. Jones, 347 U.S. 483 (1954). See Smith, 347 U.S. at 495. Id. at 496. Id. at 497."


def _partial_name(value: str) -> dict[str, object]:
    return {
        "kind": "partial",
        "plaintiff": None,
        "defendant": None,
        "subject": None,
        "partial": value,
    }


def _quote(text: str, value: str) -> dict[str, object]:
    start = text.index(value)
    return {"start": start, "end": start + len(value), "quote": value}


def _quote_last(text: str, value: str) -> dict[str, object]:
    start = text.rindex(value)
    return {"start": start, "end": start + len(value), "quote": value}


def _write_fixture(tmp_path: Path) -> Path:
    dataset = tmp_path / "primary"
    source_dir = dataset / "documents_txt"
    annotation_dir = dataset / "documents"
    source_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    source = source_dir / "example.txt"
    source.write_text(TEXT, encoding="utf-8")
    digest = hashlib.sha256(TEXT.encode("utf-8")).hexdigest()
    header = {
        "unit": "header",
        "dataset": "primary",
        "document": source.name,
        "text": {
            "path": "primary/documents_txt/example.txt",
            "sha256": digest,
            "length": len(TEXT),
        },
    }
    rows = [
        {
            "unit": "citation",
            "id": "example-o01",
            "kind": "FullCaseCitation",
            "is_root": True,
            "root_id": "example-o01",
            "locator": _quote(TEXT, "347 U.S. 483"),
        },
        {
            "unit": "citation",
            "id": "example-o02",
            "kind": "ShortCaseCitation",
            "is_root": False,
            "root_id": "example-o01",
            "cited_as": _quote(TEXT, "Smith, 347 U.S. at 495"),
            "case_name": {
                "source": {"kind": "quoted", **_quote_last(TEXT, "Smith")},
                "normalization": {"kind": "value", "value": _partial_name("Smith")},
            },
            "locator": {
                "source": {"kind": "quoted", **_quote(TEXT, "347 U.S. at 495")},
                "normalization": {
                    "kind": "value",
                    "value": {
                        "volume": 347,
                        "reporter": {
                            "short_name": "U.S.",
                            "name": "United States Supreme Court Reports",
                            "cite_type": "federal",
                            "source": "reporters",
                            "is_scotus": True,
                        },
                        "edition": "U.S.",
                    },
                },
            },
            "pin_cite": {
                **_quote(TEXT, "495"),
                "normalized": [{"first": 495, "last": 495, "kind": "page"}],
            },
        },
        {
            "unit": "citation",
            "id": "example-o03",
            "kind": "IdCitation",
            "is_root": False,
            "root_id": "example-o01",
            "cited_as": _quote(TEXT, "Id. at 496"),
            "pin_cite": {
                **_quote(TEXT, "496"),
                "normalized": [{"first": 496, "last": 496, "kind": "page"}],
            },
        },
        {
            "unit": "citation",
            "id": "example-o04",
            "kind": "IdCitation",
            "is_root": False,
            "root_id": "example-o01",
            "cited_as": _quote(TEXT, "Id. at 497"),
            "pin_cite": {
                **_quote(TEXT, "497"),
                "normalized": [{"first": 497, "last": 497, "kind": "page"}],
            },
        },
    ]
    (annotation_dir / "example.jsonl").write_text(
        "\n".join(json.dumps(row) for row in (header, *rows)) + "\n", encoding="utf-8"
    )
    return source


def _leaf_document(tmp_path: Path) -> Document:
    source = _write_fixture(tmp_path)
    roots = asyncio.run(grow_roots(Document.from_source(source)))
    return asyncio.run(grow_leaves(roots, review_leaves=False))


def test_stage_scores_use_exact_stage_checkpoints_and_annotation_spans(tmp_path: Path) -> None:
    document = _leaf_document(tmp_path)

    short = evaluation.score_short_reporter_citations(document)
    short_attribution = evaluation.score_short_reporter_attribution(document)
    short_names = evaluation.score_short_reporter_case_names(document)
    identifier = evaluation.score_id_citations(document)
    names = evaluation.score_supra_case_names(document)
    pins = evaluation.score_supra_pin_cites(document)

    assert short.stage == evaluation.SHORT_STAGE
    assert short.metrics["locator_span"].as_dict() == {"correct": 1, "total": 1, "precision": 1.0}
    assert short.metrics["locator_normalization"] == Precision(1, 1)
    assert short_names.metrics["case_name_span"] == Precision(1, 1)
    assert short_names.metrics["case_name_normalization"] == Precision(1, 1)
    assert short.metrics["pin_cite_span"].correct == short.metrics["pin_cite_span"].total == 1
    assert (
        short.metrics["pin_cite_normalization"].correct == short.metrics["pin_cite_normalization"].total == 1
    )
    assert short_attribution.stage == "29_short_reporter_attribution"
    assert (
        short_attribution.metrics["attribution"].correct
        == short_attribution.metrics["attribution"].total
        == 1
    )
    assert tuple(identifier.metrics) == ("citation_span", "pin_cite_span", "pin_cite_normalization")
    assert all(metric == Precision(2, 2) for metric in identifier.metrics.values())
    assert names.metrics["span"].total == 0
    assert names.metrics["normalization"].total == 0
    assert pins.metrics["span"] == Precision()
    assert pins.metrics["normalization"] == Precision()


def test_short_stage_scores_read_their_checkpoint_and_shared_rule_has_no_inherited_link(
    tmp_path: Path,
) -> None:
    document = _leaf_document(tmp_path)
    created = document.get_stage("28_short_reporter_citations")
    attributed = document.get_stage("29_short_reporter_attribution")

    assert created.short_reporters[0].root_id == ()
    assert evaluation.score_short_reporter_citations(created) == evaluation.score_short_reporter_citations(
        document
    )
    assert evaluation.score_short_reporter_attribution(
        attributed
    ) == evaluation.score_short_reporter_attribution(document)
    assert evaluation.score_supra_attribution_rule(document).metrics["attribution"].total == 0
    with pytest.raises(KeyError, match="Stage has not run"):
        evaluation.score_short_reporter_attribution(created)


def test_short_names_and_colocation_scores_use_their_own_checkpoints(tmp_path: Path) -> None:
    document = _leaf_document(tmp_path)
    created = document.get_stage(evaluation.SHORT_STAGE)
    grouped = document.get_stage(evaluation.SHORT_COLOCATION_STAGE)
    named = document.get_stage(evaluation.SHORT_NAME_STAGE)

    assert not created.short_reporters[0].case_name
    assert not grouped.short_reporters[0].case_name
    assert named.short_reporters[0].case_name[-1].quote == "Smith"
    assert evaluation.score_short_reporter_colocations(document).metrics == {}
    assert evaluation.score_short_reporter_case_names(named) == evaluation.score_short_reporter_case_names(
        document
    )
    with pytest.raises(KeyError, match="Stage has not run"):
        evaluation.score_short_reporter_case_names(grouped)
    score = evaluation.score_grow_leaves(document)
    assert [stage.stage for stage in score.stages[:4]] == [
        evaluation.SHORT_STAGE,
        evaluation.SHORT_COLOCATION_STAGE,
        evaluation.SHORT_NAME_STAGE,
        evaluation.SHORT_ATTRIBUTION_STAGE,
    ]
    report = evaluation.render_grow_leaves(score)
    assert "No precision score: independent short-reporter group annotations are not defined." in report


def test_short_creation_does_not_require_case_name_gold_before_the_name_stage(tmp_path: Path) -> None:
    document = _leaf_document(tmp_path)
    annotation = Path(document.source_path).parent.parent / "documents" / "example.jsonl"
    rows = [json.loads(line) for line in annotation.read_text().splitlines()]
    next(row for row in rows if row.get("kind") == "ShortCaseCitation").pop("case_name")
    annotation.write_text("".join(json.dumps(row) + "\n" for row in rows))

    score = evaluation.score_short_reporter_citations(document)
    assert tuple(score.metrics) == (
        "locator_span",
        "locator_normalization",
        "pin_cite_span",
        "pin_cite_normalization",
    )
    assert all(value == Precision(1, 1) for value in score.metrics.values())
    with pytest.raises(ValueError, match="Missing explicit field normalization gold"):
        evaluation.score_short_reporter_case_names(document)


@pytest.mark.parametrize("kind", ["ShortCaseCitation", "IdCitation", "ReferenceCitation"])
def test_source_span_summary_excludes_dummy_head_without_dropping_gold_or_creation_decisions(
    tmp_path: Path, kind: str
) -> None:
    if kind == "ReferenceCitation":
        source, annotation, header, rows = _reference_annotation_fixture(tmp_path)
        annotation.write_text("".join(json.dumps(row) + "\n" for row in [header, *rows]))
        document = asyncio.run(
            grow_leaves(asyncio.run(grow_roots(Document.from_source(source))), review_leaves=False)
        )
        citation = next(c for c in document.short_citations if isinstance(c, ReferenceCitation))
        stage_score = evaluation.score_reference_citations
    else:
        document = _leaf_document(tmp_path)
        citation = (
            document.short_reporters[0]
            if kind == "ShortCaseCitation"
            else next(c for c in document.short_citations if isinstance(c, IdCitation))
        )
        stage_score = (
            evaluation.score_short_reporter_citations
            if kind == "ShortCaseCitation"
            else evaluation.score_id_citations
        )
    before = evaluation.score_grow_leaves(document)
    original_creation = stage_score(document)
    # Only root_id determines membership in the dummy collection. No extra
    # exclusion flag or changed attribution/review outcome is required.
    after = document.replace_citation(citation.record(evaluation.SUPRA_REVIEW_STAGE).withdraw()).complete(
        evaluation.SUPRA_REVIEW_STAGE
    )
    restored = Document.model_validate_json(after.model_dump_json())
    summary = evaluation.score_grow_leaves(restored)

    for label in (kind, "all_leaves"):
        original = before.leaf_spans[label]
        current = summary.leaf_spans[label]
        assert current == FieldScore(original.correct - 1, original.predicted - 1, original.gold)
    assert stage_score(restored) == original_creation
    assert restored.get_stage(document.stage_runs[-1]) == document


def test_short_and_id_creation_both_count_missing_pin_outcomes(tmp_path: Path) -> None:
    document = _leaf_document(tmp_path)
    missing_pin_ids = {
        document.short_reporters[0].id,
        next(c.id for c in document.short_citations if isinstance(c, IdCitation)),
    }
    # Model a saved run with two missed readings and one retained Id. pin.
    data = document.model_dump(mode="python")
    for citation in data["citations"]:
        if citation["id"] in missing_pin_ids:
            citation["pin_cite"] = None
    saved = Document.model_validate(data)
    document = Document.model_validate_json(saved.model_dump_json())

    short = evaluation.score_short_reporter_citations(document)
    assert short.metrics["locator_span"] == Precision(1, 1)
    assert short.metrics["pin_cite_span"] == Precision(0, 1)
    assert short.metrics["pin_cite_normalization"] == Precision(0, 1)
    pins = evaluation.score_id_citations(document)
    assert pins.metrics["pin_cite_span"] == Precision(1, 2)
    assert pins.metrics["pin_cite_normalization"] == Precision(1, 2)
    assert all(metric.total == 0 for metric in evaluation.score_supra_pin_cites(document).metrics.values())
    workflow = evaluation.score_grow_leaves(document)
    assert workflow.leaf_spans["all_leaves"] == FieldScore(3, 3, 3)
    assert workflow.leaf_attribution["all_leaves"] == FieldScore(3, 3, 3)


def test_short_locator_and_name_stages_score_independent_normalization_envelopes(tmp_path: Path) -> None:
    document = _leaf_document(tmp_path)
    annotation = Path(document.source_path).parent.parent / "documents" / "example.jsonl"
    rows = [json.loads(line) for line in annotation.read_text().splitlines()]
    short = next(row for row in rows if row.get("kind") == "ShortCaseCitation")

    def persist():
        annotation.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    persist()
    score = evaluation.score_short_reporter_citations(document)
    assert score.metrics["locator_normalization"].correct == score.metrics["locator_normalization"].total == 1
    assert (
        evaluation.score_short_reporter_case_names(document).metrics["case_name_normalization"].correct == 1
    )
    short["locator"]["normalization"]["value"]["volume"] = 348
    short["case_name"]["normalization"]["value"]["partial"] = "Jones"
    persist()
    score = evaluation.score_short_reporter_citations(document)
    assert score.metrics["locator_normalization"].correct == 0
    assert (
        evaluation.score_short_reporter_case_names(document).metrics["case_name_normalization"].correct == 0
    )
    short["case_name"] = {
        "source": {"kind": "not_stated"},
        "normalization": {"kind": "unavailable"},
    }
    persist()
    assert (
        evaluation.score_short_reporter_case_names(document).metrics["case_name_normalization"].correct == 0
    )
    short["locator"]["normalization"] = {"kind": "unavailable"}
    persist()
    assert evaluation.score_short_reporter_citations(document).metrics["locator_normalization"].correct == 0


def test_workflow_scores_source_span_recall_and_root_links(tmp_path: Path) -> None:
    score = evaluation.score_grow_leaves(_leaf_document(tmp_path))

    spans = score.leaf_spans["all_leaves"].as_dict()
    attributions = score.leaf_attribution["all_leaves"].as_dict()
    assert spans == {
        "correct": 3,
        "predicted": 3,
        "gold": 3,
        "precision": 1.0,
        "recall": 1.0,
    }
    assert attributions == {
        "correct": 3,
        "predicted": 3,
        "gold": 3,
        "precision": 1.0,
        "recall": 1.0,
    }
    rendered = evaluation.render_grow_leaves(score, set_name="fixture")
    assert "Workflow recall uses annotated leaves" in rendered
    assert "| all_leaves | 3/3 (100.0%) | 3/3 (100.0%) |" in rendered


def test_score_run_registry_writes_leaf_json_and_markdown(tmp_path: Path) -> None:
    document = _leaf_document(tmp_path)
    run_dir = tmp_path / "run"
    (run_dir / "documents").mkdir(parents=True)
    (run_dir / "documents" / "example.txt.json").write_text(document.model_dump_json(), encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps({"status": "complete", "set": "fixture", "filings": ["example.txt"]}),
        encoding="utf-8",
    )

    reports = score_run(run_dir, workflows=("grow_leaves",))

    score_data = json.loads((run_dir / "grow_leaves.json").read_text(encoding="utf-8"))
    assert "grow_leaves" in reports
    assert score_data["set"] == "fixture"
    assert score_data["leaf_spans"]["all_leaves"]["gold"] == 3
    assert (
        (run_dir / "grow_leaves.md")
        .read_text(encoding="utf-8")
        .startswith("# Grow-leaves evaluation: fixture")
    )


def test_primary_annotations_keep_369_leaf_sites_with_289_in_scope_and_80_native_out_of_scope() -> None:
    dataset = Path(__file__).resolve().parents[2] / "mellea-lrc-datasets" / "primary" / "documents"
    if not dataset.is_dir():
        pytest.skip("Sibling mellea-lrc-datasets checkout is unavailable")

    leaf_keys = []
    out_of_scope_keys = []
    for path in dataset.glob("*.jsonl"):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        text = (dataset.parent.parent / rows[0]["text"]["path"]).read_text(encoding="utf-8")
        leaf_rows = [
            row
            for row in rows[1:]
            if row.get("unit") in {"citation", "out_of_scope_citation"} and not row.get("is_root")
        ]
        for row in leaf_rows:
            key = evaluation._gold_key(row)
            field = (
                "case_name"
                if row["kind"] == "ReferenceCitation"
                else (
                    "locator"
                    if row["kind"] in {"FullCaseCitation", "DocketCitation", "ShortCaseCitation"}
                    else "cited_as"
                )
            )
            span = evaluation._span(row[field])
            source = row[field].get("source", row[field])
            assert span is not None and text[span[0] : span[1]] == source["quote"]
            if row["unit"] == "out_of_scope_citation":
                assert row["kind"] == "ReferenceCitation"
                assert evaluation._span(row.get("pin_cite")) is None
                assert row.get("note")
                out_of_scope_keys.append(key)
            else:
                leaf_keys.append(key)
    assert len(leaf_keys) == 289
    assert len(out_of_scope_keys) == 80
    assert len(leaf_keys) + len(out_of_scope_keys) == 369


def test_stage_scores_do_not_accept_a_changed_source_or_missing_checkpoint(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path)
    roots = asyncio.run(grow_roots(Document.from_source(source)))
    leaves = asyncio.run(grow_leaves(roots, review_leaves=False))

    with pytest.raises(KeyError, match="Stage has not run"):
        evaluation.score_short_reporter_citations(roots)
    annotation = source.parent.parent / "documents" / "example.jsonl"
    rows = annotation.read_text(encoding="utf-8").splitlines()
    header = json.loads(rows[0])
    header["text"]["sha256"] = "0" * 64
    rows[0] = json.dumps(header)
    annotation.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="annotation does not match"):
        evaluation.score_grow_leaves(leaves)


def test_evaluator_rejects_a_mislabeled_dataset_header(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path)
    document = asyncio.run(
        grow_leaves(asyncio.run(grow_roots(Document.from_source(source))), review_leaves=False)
    )
    annotation = source.parent.parent / "documents" / "example.jsonl"
    rows = annotation.read_text(encoding="utf-8").splitlines()
    header = json.loads(rows[0])
    header["dataset"] = "reliable-low-profile"
    rows[0] = json.dumps(header)
    annotation.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="annotation does not match"):
        evaluation.score_grow_leaves(document)


def test_id_attribution_scores_final_rule_and_review_attachments(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path)
    roots = asyncio.run(grow_roots(Document.from_source(source)))
    created = asyncio.run(grow_leaves(roots, review_leaves=False)).get_stage(evaluation.ID_STAGE)
    calls = 0

    async def accept_then_fail(_context):
        nonlocal calls
        calls += 1
        if calls == 1:
            return LeafReviewDecision(is_citation=True, root_index=0, reason="Select matching root")
        return LeafReviewOutcome(None, failure_reason="Synthetic failed review")

    reviewed = asyncio.run(attribute_id_citations(created, reviewer=accept_then_fail))
    score = evaluation.score_id_attribution(reviewed)

    assert score.metrics["attribution"] == Precision(2, 2)


@pytest.mark.parametrize("gold_state", ["quoted", "not_stated", "unmatched", "missing"])
def test_id_creation_counts_absent_pin_outcomes_and_requires_explicit_gold(
    tmp_path: Path, gold_state: str
) -> None:
    document = _leaf_document(tmp_path)
    annotation = Path(document.source_path).parent.parent / "documents" / "example.jsonl"
    rows = [json.loads(line) for line in annotation.read_text().splitlines()]
    target = next(row for row in rows if row.get("id") == "example-o03")
    if gold_state == "not_stated":
        target["pin_cite"] = {
            "source": {"kind": "not_stated"},
            "normalization": {"kind": "unavailable"},
        }
    elif gold_state == "unmatched":
        rows.remove(target)
    elif gold_state == "missing":
        target.pop("pin_cite")
    annotation.write_text("".join(json.dumps(row) + "\n" for row in rows))
    data = document.model_dump(mode="python")
    first_id = next(c for c in document.short_citations if isinstance(c, IdCitation))
    next(c for c in data["citations"] if c["id"] == first_id.id)["pin_cite"] = None
    restored = Document.model_validate_json(Document.model_validate(data).model_dump_json())

    if gold_state == "missing":
        with pytest.raises(ValueError, match="Missing explicit field normalization gold"):
            evaluation.score_id_citations(restored)
    else:
        score = evaluation.score_id_citations(restored)
        assert tuple(score.metrics) == ("citation_span", "pin_cite_span", "pin_cite_normalization")
        expected = Precision(1 + int(gold_state == "not_stated"), 2)
        assert score.metrics["pin_cite_span"] == expected
        assert score.metrics["pin_cite_normalization"] == expected
        assert all(metric.total == 2 for metric in score.metrics.values())
        assert evaluation.score_id_citations(restored.get_stage(evaluation.ID_STAGE)) == score


def test_overlapping_reference_attribution_uses_a_gold_mention_only_once() -> None:
    source = "Alpha Beta"
    first = ReferenceCitation.from_source(source=source, span=Span(0, 5), stage="30_reference_citations")
    second = ReferenceCitation.from_source(source=source, span=Span(6, 10), stage="30_reference_citations")
    gold_row = {
        "id": "one-gold-mention",
        "kind": "ReferenceCitation",
        "is_root": False,
        "root_id": "root-1",
    }
    gold = {("ReferenceCitation", 0, len(source)): gold_row}

    aligned = evaluation._attribution_rows([first, second], gold)

    assert aligned[0] == (first, gold_row)
    assert aligned[1] == (second, None)


def _reference_annotation_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, object], list[dict[str, object]]]:
    text = "Smith v. Jones, 347 U.S. 483 (1954). Smith at 495. Smyth at 496. Smith held otherwise."
    dataset = tmp_path / "primary"
    source = dataset / "documents_txt" / "example.txt"
    annotations = dataset / "documents" / "example.jsonl"
    source.parent.mkdir(parents=True)
    annotations.parent.mkdir()
    source.write_text(text)
    named_start = text.index("Smith at")
    bare_start = text.index("Smith held")
    header = {
        "unit": "header",
        "dataset": "primary",
        "document": source.name,
        "text": {
            "path": "primary/documents_txt/example.txt",
            "length": len(text),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        },
    }
    rows = [
        {
            "unit": "citation",
            "id": "root",
            "kind": "FullCaseCitation",
            "is_root": True,
            "root_id": "root",
            "locator": _quote(text, "347 U.S. 483"),
        },
        {
            "unit": "citation",
            "id": "pinned-reference",
            "kind": "ReferenceCitation",
            "is_root": False,
            "root_id": "root",
            "case_name": {
                "start": named_start,
                "end": named_start + 5,
                "quote": "Smith",
                "normalized": _partial_name("Smith"),
            },
            "pin_cite": {
                **_quote(text, "495"),
                "normalized": [{"first": 495, "last": 495, "kind": "page"}],
            },
        },
        {
            "unit": "out_of_scope_citation",
            "id": "bare-reference",
            "kind": "ReferenceCitation",
            "is_root": False,
            "root_id": "root",
            "case_name": {"start": bare_start, "end": bare_start + 5, "quote": "Smith"},
            "note": "Bare name has no adjacent explicit pinpoint; outside supported reference citation shape.",
        },
    ]
    return source, annotations, header, rows


def test_reference_workflow_uses_native_scope_and_preserves_out_of_scope_annotations(tmp_path: Path) -> None:
    source, annotations, header, rows = _reference_annotation_fixture(tmp_path)
    text = source.read_text()
    annotation_text = "".join(json.dumps(row) + "\n" for row in [header, *rows])
    annotations.write_text(annotation_text)
    document = asyncio.run(grow_roots(Document.from_source(source)))
    document = asyncio.run(grow_leaves(document, review_leaves=False))

    assert {row["id"] for row in citation_annotations(document)} == {"root", "pinned-reference"}
    assert {row["id"] for row in evaluation._gold(document).values()} == {"root", "pinned-reference"}
    retained = next(row for row in rows if row["id"] == "bare-reference")
    assert retained["unit"] == "out_of_scope_citation"
    assert retained["kind"] == "ReferenceCitation"
    assert retained["note"]
    creation_score = evaluation.score_reference_citations(document)
    assert tuple(creation_score.metrics) == (
        "case_name_span",
        "case_name_normalization",
        "pin_cite_span",
        "pin_cite_normalization",
    )
    assert creation_score.metrics["case_name_span"].as_dict() == {
        "correct": 1,
        "total": 1,
        "precision": 1.0,
    }
    assert creation_score.metrics["case_name_normalization"] == Precision(1, 1)
    assert creation_score.metrics["pin_cite_span"] == Precision(1, 1)
    assert creation_score.metrics["pin_cite_normalization"] == Precision(1, 1)
    report = evaluation.render_reference_citations(creation_score)
    for label in creation_score.metrics:
        assert f"| {label} |" in report
    assert "| span |" not in report
    assert "| normalization |" not in report
    created = document.get_stage("30_reference_citations")
    attributed = document.get_stage("31_reference_attribution")
    assert next(c for c in created.short_citations if isinstance(c, ReferenceCitation)).root_id == ()
    assert evaluation.score_reference_citations(created) == creation_score
    assert evaluation.score_reference_attribution(attributed) == evaluation.score_reference_attribution(
        document
    )
    assert evaluation.score_reference_attribution(document).metrics["attribution"] == Precision(1, 1)
    with pytest.raises(KeyError, match="Stage has not run"):
        evaluation.score_reference_attribution(created)
    for scorer in (evaluation.score_supra_case_names, evaluation.score_supra_pin_cites):
        assert all(metric.total == 0 for metric in scorer(document).metrics.values())
    assert evaluation.score_supra_attribution_rule(document).metrics["attribution"] == Precision()
    reference = evaluation.score_grow_leaves(document).leaf_spans["ReferenceCitation"]
    assert (reference.correct, reference.predicted, reference.gold) == (1, 1, 1)
    assert annotations.read_text() == annotation_text
    assert (
        next(
            json.loads(line)
            for line in annotations.read_text().splitlines()
            if json.loads(line).get("id") == "bare-reference"
        )
        == retained
    )

    # A missed, independently annotated pinpoint stays in the denominator.
    # The scorer never decides scope by whether the detector found the site.
    rows.append(
        {
            "unit": "citation",
            "id": "missed-reference",
            "kind": "ReferenceCitation",
            "is_root": False,
            "root_id": "root",
            "case_name": _quote(text, "Smyth"),
            "pin_cite": _quote(text, "496"),
        }
    )
    annotations.write_text("".join(json.dumps(row) + "\n" for row in [header, *rows]))
    assert evaluation.score_grow_leaves(document).leaf_spans["ReferenceCitation"].gold == 2


@pytest.mark.parametrize("pin_field", [None, {"normalized": [{"first": 495, "last": 495, "kind": "page"}]}])
def test_in_scope_reference_without_a_quoted_pin_raises_instead_of_disappearing_from_gold(
    tmp_path: Path, pin_field: dict[str, object] | None
) -> None:
    source, annotations, header, rows = _reference_annotation_fixture(tmp_path)
    bare = next(row for row in rows if row["id"] == "bare-reference")
    bare["unit"] = "citation"
    if pin_field is not None:
        bare["pin_cite"] = pin_field
    annotations.write_text("".join(json.dumps(row) + "\n" for row in [header, *rows]))
    document = asyncio.run(grow_roots(Document.from_source(source)))
    document = asyncio.run(grow_leaves(document, review_leaves=False))
    before = annotations.read_text()

    assert "bare-reference" in {row["id"] for row in citation_annotations(document)}
    for score in (evaluation._gold, evaluation.score_reference_citations, evaluation.score_grow_leaves):
        with pytest.raises(ValueError, match="bare-reference") as error:
            score(document)
        assert "out_of_scope_citation" in str(error.value)
    assert annotations.read_text() == before


@pytest.mark.parametrize("missing_field", ["case_name", "pin_cite"])
def test_reference_creation_requires_independent_normalization_for_each_predicted_field(
    tmp_path: Path, missing_field: str
) -> None:
    source, annotations, header, rows = _reference_annotation_fixture(tmp_path)
    reference = next(row for row in rows if row["id"] == "pinned-reference")
    reference[missing_field].pop("normalized")
    annotations.write_text("".join(json.dumps(row) + "\n" for row in [header, *rows]))
    document = asyncio.run(grow_roots(Document.from_source(source)))
    document = asyncio.run(grow_leaves(document, review_leaves=False))

    with pytest.raises(ValueError, match="normalization"):
        evaluation.score_reference_citations(document)


@pytest.mark.parametrize("missing_field", ["locator", "case_name", "pin_cite"])
def test_short_stages_require_independent_normalization_for_each_predicted_field(
    tmp_path: Path, missing_field: str
) -> None:
    document = _leaf_document(tmp_path)
    annotations = Path(document.source_path).parent.parent / "documents" / "example.jsonl"
    rows = [json.loads(line) for line in annotations.read_text().splitlines()]
    short = next(row for row in rows if row.get("kind") == "ShortCaseCitation")
    short[missing_field].pop("normalized" if missing_field == "pin_cite" else "normalization")
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows))

    scorer = (
        evaluation.score_short_reporter_case_names
        if missing_field == "case_name"
        else evaluation.score_short_reporter_citations
    )
    with pytest.raises(ValueError, match="normalization"):
        scorer(document)


def test_unmatched_short_prediction_counts_in_every_emitted_field_metric(tmp_path: Path) -> None:
    document = _leaf_document(tmp_path)
    annotations = Path(document.source_path).parent.parent / "documents" / "example.jsonl"
    rows = [json.loads(line) for line in annotations.read_text().splitlines()]
    rows = [row for row in rows if row.get("kind") != "ShortCaseCitation"]
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows))

    score = evaluation.score_short_reporter_citations(document)

    assert tuple(score.metrics) == (
        "locator_span",
        "locator_normalization",
        "pin_cite_span",
        "pin_cite_normalization",
    )
    assert all(metric == Precision(0, 1) for metric in score.metrics.values())
    names = evaluation.score_short_reporter_case_names(document)
    assert tuple(names.metrics) == ("case_name_span", "case_name_normalization")
    assert all(metric == Precision(0, 1) for metric in names.metrics.values())


def test_unmatched_reference_prediction_counts_in_every_emitted_field_metric(tmp_path: Path) -> None:
    source, annotations, header, rows = _reference_annotation_fixture(tmp_path)
    rows = [row for row in rows if row["id"] != "pinned-reference"]
    annotations.write_text("".join(json.dumps(row) + "\n" for row in [header, *rows]))
    document = asyncio.run(grow_roots(Document.from_source(source)))
    document = asyncio.run(grow_leaves(document, review_leaves=False))

    score = evaluation.score_reference_citations(document)

    assert tuple(score.metrics) == (
        "case_name_span",
        "case_name_normalization",
        "pin_cite_span",
        "pin_cite_normalization",
    )
    assert all(metric == Precision(0, 1) for metric in score.metrics.values())


@pytest.mark.parametrize("field", ["case_name", "pin_cite"])
@pytest.mark.parametrize("gold_state", ["quoted", "not_stated", "unmatched"])
def test_short_stages_count_absence_predictions_against_explicit_gold(
    tmp_path: Path, field: str, gold_state: str
) -> None:
    document = _leaf_document(tmp_path)
    annotations = Path(document.source_path).parent.parent / "documents" / "example.jsonl"
    rows = [json.loads(line) for line in annotations.read_text().splitlines()]
    short = next(row for row in rows if row.get("kind") == "ShortCaseCitation")
    if gold_state == "not_stated":
        short[field] = {
            "source": {"kind": "not_stated"},
            "normalization": {"kind": "unavailable"},
        }
    elif gold_state == "unmatched":
        rows.remove(short)
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows))
    data = document.model_dump(mode="python")
    next(c for c in data["citations"] if c["id"] == document.short_reporters[0].id)[field] = (
        () if field == "case_name" else None
    )
    saved = Document.model_validate(data)
    if field == "case_name":
        assert saved.short_reporters[0].get_case_name().kind.value == "not_stated"

    scorer = (
        evaluation.score_short_reporter_case_names
        if field == "case_name"
        else evaluation.score_short_reporter_citations
    )
    score = scorer(saved)

    expected = Precision(int(gold_state == "not_stated"), 1)
    assert score.metrics[f"{field}_span"] == expected
    assert score.metrics[f"{field}_normalization"] == expected


@pytest.mark.parametrize("field", ["case_name", "pin_cite"])
def test_short_stages_require_explicit_gold_for_absence_predictions(tmp_path: Path, field: str) -> None:
    document = _leaf_document(tmp_path)
    annotations = Path(document.source_path).parent.parent / "documents" / "example.jsonl"
    rows = [json.loads(line) for line in annotations.read_text().splitlines()]
    short = next(row for row in rows if row.get("kind") == "ShortCaseCitation")
    short.pop(field)
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows))
    data = document.model_dump(mode="python")
    next(c for c in data["citations"] if c["id"] == document.short_reporters[0].id)[field] = (
        () if field == "case_name" else None
    )
    saved = Document.model_validate(data)

    scorer = (
        evaluation.score_short_reporter_case_names
        if field == "case_name"
        else evaluation.score_short_reporter_citations
    )
    with pytest.raises(ValueError):
        scorer(saved)


@pytest.mark.parametrize("gold_state", ["quoted", "unmatched"])
def test_reference_creation_counts_pin_absence_as_an_incorrect_prediction(
    tmp_path: Path, gold_state: str
) -> None:
    source, annotations, header, rows = _reference_annotation_fixture(tmp_path)
    if gold_state == "unmatched":
        rows = [row for row in rows if row["id"] != "pinned-reference"]
    annotations.write_text("".join(json.dumps(row) + "\n" for row in [header, *rows]))
    document = asyncio.run(grow_roots(Document.from_source(source)))
    document = asyncio.run(grow_leaves(document, review_leaves=False))
    data = document.model_dump(mode="python")
    reference = next(c for c in document.short_citations if isinstance(c, ReferenceCitation))
    next(c for c in data["citations"] if c["id"] == reference.id)["pin_cite"] = None
    saved = Document.model_validate(data)

    score = evaluation.score_reference_citations(saved)

    assert score.metrics["pin_cite_span"] == Precision(0, 1)
    assert score.metrics["pin_cite_normalization"] == Precision(0, 1)


def test_short_name_stage_counts_a_name_prediction_against_explicit_absence_as_incorrect(
    tmp_path: Path,
) -> None:
    document = _leaf_document(tmp_path)
    annotations = Path(document.source_path).parent.parent / "documents" / "example.jsonl"
    rows = [json.loads(line) for line in annotations.read_text().splitlines()]
    short = next(row for row in rows if row.get("kind") == "ShortCaseCitation")
    short["case_name"] = {
        "source": {"kind": "not_stated"},
        "normalization": {"kind": "unavailable"},
    }
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows))

    score = evaluation.score_short_reporter_case_names(document)

    assert score.metrics["case_name_span"] == Precision(0, 1)
    assert score.metrics["case_name_normalization"] == Precision(0, 1)


@pytest.mark.parametrize(
    "case_name",
    [
        {
            "source": {"kind": "not_stated"},
            "normalization": {"kind": "value", "value": _partial_name("Smith")},
        },
        {"source": {"kind": "unknown"}, "normalization": {"kind": "unavailable"}},
        {"source": {"kind": "not_stated"}},
    ],
)
@pytest.mark.parametrize("has_name_reading", [True, False])
def test_short_name_stage_rejects_incomplete_or_inconsistent_explicit_name_absence(
    tmp_path: Path, case_name: dict[str, object], has_name_reading: bool
) -> None:
    document = _leaf_document(tmp_path)
    annotations = Path(document.source_path).parent.parent / "documents" / "example.jsonl"
    rows = [json.loads(line) for line in annotations.read_text().splitlines()]
    short = next(row for row in rows if row.get("kind") == "ShortCaseCitation")
    short["case_name"] = case_name
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if not has_name_reading:
        data = document.model_dump(mode="python")
        next(c for c in data["citations"] if c["id"] == document.short_reporters[0].id)["case_name"] = ()
        document = Document.model_validate(data)

    with pytest.raises(ValueError):
        evaluation.score_short_reporter_case_names(document)


def test_id_creation_counts_unmatched_readings_in_span_and_normalization(tmp_path: Path) -> None:
    document = _leaf_document(tmp_path)
    annotations = Path(document.source_path).parent.parent / "documents" / "example.jsonl"
    rows = [json.loads(line) for line in annotations.read_text().splitlines()]
    rows = [row for row in rows if row.get("id") != "example-o03"]
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows))

    score = evaluation.score_id_citations(document)

    assert all(metric == Precision(1, 2) for metric in score.metrics.values())


def test_id_creation_requires_independent_normalization_for_matched_readings(tmp_path: Path) -> None:
    document = _leaf_document(tmp_path)
    annotations = Path(document.source_path).parent.parent / "documents" / "example.jsonl"
    rows = [json.loads(line) for line in annotations.read_text().splitlines()]
    next(row for row in rows if row.get("id") == "example-o03")["pin_cite"].pop("normalized")
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="normalization"):
        evaluation.score_id_citations(document)


@pytest.mark.parametrize("gold_state", ["unmatched", "missing_normalization"])
def test_leaf_name_stage_counts_emitted_readings_and_requires_their_matched_gold(
    tmp_path: Path, gold_state: str
) -> None:
    document = _leaf_document(tmp_path).get_stage(evaluation.SUPRA_STAGE)
    short = document.short_reporters[0]
    # Model a saved run with a name-stage reading, preserving its earlier
    # creation reading so the scorer must select the field emitted at the name stage.
    reread = short.record(evaluation.SUPRA_NAME_STAGE).with_case_name(document.text, short.case_name[-1].span)
    document = document.replace_citation(reread).complete(evaluation.SUPRA_NAME_STAGE)
    annotations = Path(document.source_path).parent.parent / "documents" / "example.jsonl"
    rows = [json.loads(line) for line in annotations.read_text().splitlines()]
    if gold_state == "unmatched":
        rows = [row for row in rows if row.get("kind") != "ShortCaseCitation"]
    else:
        next(row for row in rows if row.get("kind") == "ShortCaseCitation")["case_name"].pop("normalization")
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows))

    if gold_state == "missing_normalization":
        with pytest.raises(ValueError, match="normalization"):
            evaluation.score_supra_case_names(document)
    else:
        score = evaluation.score_supra_case_names(document)
        assert score.metrics["span"] == Precision(0, 1)
        assert score.metrics["normalization"] == Precision(0, 1)


@pytest.mark.parametrize(
    "citation_kind,field", [("SupraCitation", "case_name"), ("SupraCitation", "pin_cite")]
)
@pytest.mark.parametrize("gold_state", ["quoted", "not_stated", "unmatched", "missing"])
def test_shared_leaf_readers_count_null_outcomes_without_a_written_node(
    tmp_path: Path, citation_kind: str, field: str, gold_state: str
) -> None:
    text = "Smith, supra, at 495"
    source = tmp_path / "primary" / "documents_txt" / "example.txt"
    annotation = source.parent.parent / "documents" / "example.jsonl"
    source.parent.mkdir(parents=True)
    annotation.parent.mkdir()
    source.write_text(text)
    header = {
        "unit": "header",
        "dataset": "primary",
        "document": source.name,
        "text": {
            "path": "primary/documents_txt/example.txt",
            "length": len(text),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        },
    }
    row = {
        "unit": "citation",
        "id": "example-o01",
        "kind": citation_kind,
        "is_root": False,
        "root_id": "root",
        "cited_as": _quote(text, text),
        field: (
            {**_quote(text, "Smith"), "normalized": _partial_name("Smith")}
            if field == "case_name"
            else {**_quote(text, "495"), "normalized": [{"first": 495, "last": 495, "kind": "page"}]}
        ),
    }
    if gold_state == "not_stated":
        row[field] = {"source": {"kind": "not_stated"}, "normalization": {"kind": "unavailable"}}
    elif gold_state == "missing":
        row.pop(field)
    rows = [header] if gold_state == "unmatched" else [header, row]
    annotation.write_text("".join(json.dumps(r) + "\n" for r in rows))
    creation_stage = evaluation.SUPRA_STAGE
    citation = SupraCitation.from_source(source=text, span=Span(0, len(text)), stage=creation_stage)
    document = Document.from_source(source).add_citation(citation).complete(creation_stage)
    stage = evaluation.SUPRA_NAME_STAGE if field == "case_name" else evaluation.SUPRA_PIN_STAGE
    document = document.complete(stage)
    scorer = evaluation.score_supra_case_names if field == "case_name" else evaluation.score_supra_pin_cites
    # No reading node exists, just the completed stage and its null outcome.
    assert not any(node.stage == stage for node in document.citations[0].nodes)
    if gold_state == "missing":
        with pytest.raises(ValueError, match="Missing explicit field normalization gold"):
            scorer(document)
    else:
        score = scorer(Document.model_validate_json(document.model_dump_json()))
        expected = Precision(int(gold_state == "not_stated"), 1)
        assert score.metrics["span"] == expected
        assert score.metrics["normalization"] == expected
