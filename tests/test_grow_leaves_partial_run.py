"""Completed leaf prefixes remain resumable, source-grounded, and scoreable."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path

import pytest

from evaluations import grow_leaves as evaluation
from evaluations import run_grow_leaves as runner
from evaluations.score_types import FieldScore, Precision
from mellea_lrc.api import Document, grow_roots
from mellea_lrc.extraction.leaf_attribution_review.reviewer import IvrLeafReviewer
from mellea_lrc.model.citations import AttributionResult, IdCitation, ReferenceCitation, SupraCitation
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID
from mellea_lrc.workflows.grow_leaves import grow_leaves
from tests.test_grow_leaves_score_api import _leaf_document, _write_fixture

PREFIX = (
    "28_short_reporter_citations",
    "28.1_short_reporter_colocations",
    "28.2_short_reporter_case_names",
    "29_short_reporter_attribution",
    "30_reference_citations",
    "31_reference_attribution",
)
TEXT = (
    "Smith v. Jones, 347 U.S. 483 (1954). See Smith, 347 U.S. at 495. "
    "Wrong, 347 U.S. at 499. Smith, supra, at 500. Id. at 501. See Smith at 502."
)


@pytest.fixture(autouse=True)
def prohibit_leaf_reviewers(monkeypatch) -> None:
    def reject(*_args, **_kwargs):
        pytest.fail("A rule-only run must not construct an IvrLeafReviewer")

    monkeypatch.setattr(IvrLeafReviewer, "from_profile", classmethod(reject))
    monkeypatch.setattr(IvrLeafReviewer, "__init__", reject)


def _saved_input(tmp_path: Path, document: Document, *, set_name: str = "primary") -> Path:
    documents = tmp_path / "input-run" / "documents"
    documents.mkdir(parents=True)
    filename = Path(document.source_path).name
    (documents / f"{filename}.json").write_text(document.model_dump_json(), encoding="utf-8")
    (documents.parent / "run.json").write_text(
        json.dumps({"set": set_name, "status": "complete", "filings": [filename]}),
        encoding="utf-8",
    )
    return documents


def _source_roots(tmp_path: Path) -> Document:
    source = tmp_path / "filing.txt"
    source.write_text(TEXT, encoding="utf-8")
    return asyncio.run(grow_roots(Document.from_source(source)))


def _read_output(run_dir: Path, filename: str = "filing.txt") -> Document:
    return Document.model_validate_json(
        (run_dir / "documents" / f"{filename}.json").read_text(encoding="utf-8")
    )


def test_rule_only_stop_saves_creation_and_attribution_checkpoints_and_field_history(
    tmp_path: Path, monkeypatch
) -> None:
    roots = _source_roots(tmp_path)
    inputs = _saved_input(tmp_path, roots)
    monkeypatch.setattr(runner, "_RESULTS_ROOT", tmp_path / "runs")
    saved = []
    write = runner._write

    def capture(path: Path, text: str) -> None:
        write(path, text)
        if path.parent.name == "documents":
            saved.append(Document.model_validate_json(text))

    monkeypatch.setattr(runner, "_write", capture)
    run_dir = asyncio.run(
        runner.run(inputs, input_stage="10_roots", review_leaves=False, stop_after=PREFIX[-1])
    )
    document = _read_output(run_dir)

    assert document.stage_runs == (*roots.stage_runs, *PREFIX)
    assert tuple(checkpoint.stage_runs[-1] for checkpoint in saved) == PREFIX
    assert document.get_stage("10_roots") == roots
    for stage, checkpoint in zip(PREFIX, saved, strict=True):
        assert document.get_stage(stage) == checkpoint
        assert Document.model_validate_json(checkpoint.model_dump_json()) == checkpoint
        for citation in checkpoint.citations:
            citation.validate_source(TEXT)
    assert Document.model_validate_json(document.model_dump_json()).model_dump() == document.model_dump()

    assert len(document.short_reporters) == 2
    attached, unresolved = document.short_reporters
    assert attached.case_name[-1].quote == "Smith"
    assert attached.pin_cite[-1].quote == "495"
    assert attached.root_id[-1].value == roots.roots[0].id
    assert unresolved.case_name[-1].quote == "Wrong"
    assert unresolved.pin_cite[-1].quote == "499"
    assert unresolved.root_id[-1].value == WITHDRAWN_ROOT_ID
    assert unresolved.attributions[-1].result is AttributionResult.UNRESOLVED
    for short in document.short_reporters:
        assert tuple(node.stage for node in short.nodes) == (PREFIX[0], *PREFIX[2:4])
        assert short.reviews == ()
        created = document.get_stage(PREFIX[0]).short_reporters[document.short_reporters.index(short)]
        assert created.short_locator == short.short_locator
        assert created.case_name == ()
        assert (
            document.get_stage(PREFIX[2]).short_reporters[document.short_reporters.index(short)].case_name
            == short.case_name
        )
        assert created.pin_cite == short.pin_cite
        assert created.root_id == ()

    assert not any(isinstance(c, (SupraCitation, IdCitation)) for c in document.short_citations)
    reference = next(c for c in document.short_citations if isinstance(c, ReferenceCitation))
    assert reference.reference_name[-1].quote == "Smith"
    assert reference.case_name[-1].quote == "Smith"
    assert reference.pin_cite[-1].quote == "502"
    assert tuple(node.stage for node in reference.nodes) == PREFIX[4:]
    assert reference.root_id[-1].value == roots.roots[0].id
    assert reference.attributions[-1].result is AttributionResult.ATTACHED
    assert reference.reviews == ()
    created_reference = document.get_stage(PREFIX[4]).short_citations[-1]
    assert created_reference.reference_name == reference.reference_name
    assert created_reference.case_name == reference.case_name
    assert created_reference.pin_cite == reference.pin_cite
    assert created_reference.root_id == created_reference.attributions == created_reference.reviews == ()
    assert all(
        field.node_id == created_reference.nodes[0].id
        for field in (
            created_reference.reference_name[-1],
            created_reference.case_name[-1],
            created_reference.pin_cite[-1],
        )
    )

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "complete"
    assert record["review_leaves"] is False
    assert record["stop_after"] == PREFIX[-1]


@pytest.mark.parametrize(
    "set_name",
    [
        "primary",
        "hallucination-set-1",
        "hallucination-set-2",
        "reliable-high-profile",
        "reliable-low-profile",
    ],
)
def test_interrupted_prefix_resume_reuses_the_saved_stop_configuration(
    tmp_path: Path, monkeypatch, set_name: str
) -> None:
    inputs = _saved_input(tmp_path, _source_roots(tmp_path), set_name=set_name)
    monkeypatch.setattr(runner, "_RESULTS_ROOT", tmp_path / "runs")
    reference_stage = dict(runner._STAGES)[PREFIX[-1]]
    interrupted = True
    calls = 0

    async def reference(current: Document, *, review: bool = True) -> Document:
        nonlocal calls
        calls += 1
        assert review is False
        if interrupted:
            raise RuntimeError("Synthetic attribution interruption")
        return await reference_stage(current, review=review)

    monkeypatch.setattr(runner, "attribute_reference_citations", reference)
    monkeypatch.setattr(
        runner,
        "_STAGES",
        tuple((stage, reference if stage == PREFIX[-1] else call) for stage, call in runner._STAGES),
    )
    with pytest.raises(RuntimeError, match="Synthetic attribution interruption"):
        asyncio.run(runner.run(inputs, input_stage="10_roots", review_leaves=False, stop_after=PREFIX[-1]))
    run_dir = next((tmp_path / "runs" / set_name).iterdir())
    before = _read_output(run_dir)
    assert before.stage_runs[-1] == PREFIX[-2]
    failed_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert failed_record["status"] == "failed"
    assert failed_record["set"] == set_name
    assert failed_record["stop_after"] == PREFIX[-1]

    interrupted = False
    unused_results_root = tmp_path / "unused-results"
    monkeypatch.setattr(runner, "_RESULTS_ROOT", unused_results_root)
    assert asyncio.run(runner.run(None, resume_run=run_dir)) == run_dir
    after = _read_output(run_dir)
    assert calls == 2
    assert after.stage_runs[-len(PREFIX) :] == PREFIX
    assert after.get_stage(PREFIX[-2]) == before
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "complete"
    assert record["set"] == set_name
    assert record["input_stage"] == "10_roots"
    assert record["input_sha256"] == failed_record["input_sha256"]
    assert record["stop_after"] == PREFIX[-1]
    assert record["review_leaves"] is False
    assert "error" not in record
    assert not unused_results_root.exists()


@pytest.mark.parametrize(
    ("input_stage", "stop_after"),
    [
        ("32_id_citations", "28_short_reporter_citations"),
        ("32_id_citations", "32_id_citations"),
        ("10_roots", "38_supra_attribution_llm"),
        ("10_roots", "unknown_stage"),
    ],
)
def test_invalid_stop_is_rejected_before_creating_a_run(
    tmp_path: Path, monkeypatch, input_stage: str, stop_after: str
) -> None:
    document = _source_roots(tmp_path)
    if input_stage != "10_roots":
        document = asyncio.run(grow_leaves(document, review_leaves=False)).get_stage(input_stage)
    inputs = _saved_input(tmp_path, document)
    results_root = tmp_path / "runs"
    monkeypatch.setattr(runner, "_RESULTS_ROOT", results_root)

    with pytest.raises(ValueError, match=r"[Ss]top"):
        asyncio.run(runner.run(inputs, input_stage=input_stage, review_leaves=False, stop_after=stop_after))

    assert not results_root.exists()


def test_cli_passes_the_requested_stop_stage(tmp_path: Path, monkeypatch, capsys) -> None:
    inputs = tmp_path / "documents"
    calls = []

    async def fake_run(input_documents, **options):
        calls.append((input_documents, options))
        return tmp_path / "result"

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_grow_leaves", "--input-documents", str(inputs), "--rule-only", "--stop-after", PREFIX[-1]],
    )
    runner.main()

    assert calls[0][0] == inputs.resolve()
    assert calls[0][1]["review_leaves"] is False
    assert calls[0][1]["stop_after"] == PREFIX[-1]
    assert str(tmp_path / "result") in capsys.readouterr().out


def test_prefix_scoring_and_default_run_report_include_only_completed_leaf_stages(
    tmp_path: Path, monkeypatch
) -> None:
    roots = asyncio.run(grow_roots(Document.from_source(_write_fixture(tmp_path))))
    inputs = _saved_input(tmp_path, roots)
    monkeypatch.setattr(runner, "_RESULTS_ROOT", tmp_path / "runs")
    run_dir = asyncio.run(
        runner.run(inputs, input_stage="10_roots", review_leaves=False, stop_after=PREFIX[-1])
    )
    document = _read_output(run_dir, "example.txt")
    score = evaluation.score_grow_leaves(document)

    assert tuple(stage.stage for stage in score.stages) == PREFIX
    assert score.leaf_spans["all_leaves"] == FieldScore(1, 1, 1)
    assert not score.leaf_attribution
    assert score.stages[3] == evaluation.score_short_reporter_attribution(document)
    assert score.stages[3].metrics["attribution"] == Precision(1, 1)
    assert score.stages[5] == evaluation.score_reference_attribution(document)
    assert score.stages[5].metrics["attribution"] == Precision()
    assert "IdCitation" not in score.leaf_spans
    assert "SupraCitation" not in score.leaf_spans
    report = evaluation.render_grow_leaves(score, set_name="fixture")
    assert "## Leaf source spans" in report
    assert "## Leaf attribution\n" not in report
    for stage in PREFIX:
        assert stage in report
    for stage in tuple(evaluation.GROW_LEAVES_SCORERS)[len(PREFIX) :]:
        assert stage not in report

    score_module = importlib.import_module("evaluations.score_run")
    # Isolate default leaf selection: these source roots have no validation run.
    monkeypatch.setattr(score_module, "_WORKFLOWS", {"grow_leaves": score_module._WORKFLOWS["grow_leaves"]})
    reports = score_module.score_run(run_dir)
    assert tuple(reports) == ("grow_leaves",)
    saved = json.loads((run_dir / "grow_leaves.json").read_text(encoding="utf-8"))
    assert tuple(stage["stage"] for stage in saved["stages"]) == PREFIX
    assert not saved.get("leaf_attribution")
    assert (run_dir / "grow_leaves.md").read_text(encoding="utf-8") == reports["grow_leaves"]


def test_completed_workflow_retains_full_span_and_attribution_scoring(tmp_path: Path) -> None:
    document = _leaf_document(tmp_path)
    score = evaluation.score_grow_leaves(document)

    expected = tuple(stage for stage in evaluation.GROW_LEAVES_SCORERS if not stage.endswith("_llm"))
    assert tuple(stage.stage for stage in score.stages) == expected
    assert score.leaf_spans["all_leaves"] == FieldScore(3, 3, 3)
    assert score.leaf_attribution["all_leaves"] == FieldScore(3, 3, 3)
    assert "## Leaf attribution\n" in evaluation.render_grow_leaves(score)
