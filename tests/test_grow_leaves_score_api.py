"""Source-grounded leaf evaluation and score_run integration tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from evaluations import grow_leaves as evaluation
from evaluations.score_run import score_run
from mellea_lrc.api import Document, grow_roots
from mellea_lrc.extraction.id_attribution_llm import review_id_attributions
from mellea_lrc.extraction.leaf_attribution_llm.reviewer import LeafReviewOutcome
from mellea_lrc.model.citations import LeafReviewDecision, ReferenceCitation
from mellea_lrc.model.span import Span
from mellea_lrc.workflows.grow_leaves import grow_leaves

TEXT = (
    "Smith v. Jones, 347 U.S. 483 (1954). "
    "See Smith, 347 U.S. at 495. Id. at 496. Id. at 497."
)


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
            "case_name": _quote_last(TEXT, "Smith"),
            "locator": _quote(TEXT, "347 U.S. at 495"),
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
    identifier = evaluation.score_id_citations(document)
    names = evaluation.score_leaf_case_names(document)
    pins = evaluation.score_leaf_pin_cites(document)

    assert short.stage == evaluation.SHORT_STAGE
    assert short.metrics["span"].as_dict() == {"correct": 1, "total": 1, "precision": 1.0}
    assert identifier.metrics["span"].correct == identifier.metrics["span"].total == 2
    assert names.metrics["span"].correct == names.metrics["span"].total == 1
    assert names.metrics["normalization"].total == 0  # Gold supplies no independent short-name normalization.
    assert pins.metrics["span"].correct == pins.metrics["span"].total == 3
    assert pins.metrics["normalization"].correct == pins.metrics["normalization"].total == 3


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
    assert (run_dir / "grow_leaves.md").read_text(encoding="utf-8").startswith(
        "# Grow-leaves evaluation: fixture"
    )


def test_primary_annotation_source_span_denominator_is_367() -> None:
    dataset = Path(__file__).resolve().parents[2] / "mellea-lrc-datasets" / "primary" / "documents"
    if not dataset.is_dir():
        pytest.skip("Sibling mellea-lrc-datasets checkout is unavailable")

    leaf_keys = []
    for path in dataset.glob("*.jsonl"):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        text = (dataset.parent.parent / rows[0]["text"]["path"]).read_text(encoding="utf-8")
        leaf_rows = [
            row for row in rows[1:] if row.get("unit") == "citation" and not row.get("is_root")
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
            assert span is not None and text[span[0] : span[1]] == row[field]["quote"]
            leaf_keys.append(key)
    assert len(leaf_keys) == 367


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


def test_id_review_stage_scorer_excludes_inherited_rule_attachment(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path)
    roots = asyncio.run(grow_roots(Document.from_source(source)))
    rule_result = asyncio.run(grow_leaves(roots, review_leaves=False))
    calls = 0

    async def accept_then_fail(_context):
        nonlocal calls
        calls += 1
        if calls == 1:
            return LeafReviewDecision(is_citation=True, root_index=0, reason="Select matching root")
        return LeafReviewOutcome(None, failure_reason="Synthetic failed review")

    reviewed = asyncio.run(review_id_attributions(rule_result, reviewer=accept_then_fail))
    score = evaluation.score_id_attribution_llm(reviewed)

    assert score.metrics["attribution"].correct == 1
    assert score.metrics["attribution"].total == 1


def test_overlapping_reference_attribution_uses_a_gold_mention_only_once() -> None:
    source = "Alpha Beta"
    first = ReferenceCitation.from_source(source=source, span=Span(0, 5), stage="31_reference_citations")
    second = ReferenceCitation.from_source(source=source, span=Span(6, 10), stage="31_reference_citations")
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
