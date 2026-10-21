"""Checkpoint-resume behavior for the leaf workflow runner."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from evaluations import run_grow_leaves as runner
from mellea_lrc.api import (
    Document,
    attribute_reference_citations,
    attribute_short_reporter_citations,
    find_reference_citations,
    grow_roots,
    resolve_short_reporter_case_names,
    resolve_short_reporter_colocations,
)
from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.extraction.find_short_reporter_citations import find_short_reporter_citations
from mellea_lrc.extraction.predict_short_reporter_fields import predict_short_reporter_fields
from mellea_lrc.field_extraction.models import MLFieldPrediction
from mellea_lrc.parsing.span import Span

SHORT_PREFIX = (
    "grow_leaves.grow_short_reporter_leaves.find_short_reporter_citations",
    "grow_leaves.grow_short_reporter_leaves.resolve_colocations",
    "grow_leaves.grow_short_reporter_leaves.predict_fields",
    "grow_leaves.grow_short_reporter_leaves.resolve_case_names",
    "grow_leaves.grow_short_reporter_leaves.attribute_citations",
)


@pytest.fixture(autouse=True)
def _mock_native_source_boundary(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "MELLEA_LRC_FIELD_EXTRACTION_ML_ENABLED=false\nMELLEA_LRC_FIELD_EXTRACTION_CASE_NAME=RULE\nMELLEA_LRC_FIELD_EXTRACTION_COURT=RULE\nMELLEA_LRC_FIELD_EXTRACTION_DATE=RULE\nMELLEA_LRC_FIELD_EXTRACTION_PIN_CITE=RULE\n"
    )
    monkeypatch.chdir(tmp_path)
    # These fixtures isolate workflow persistence; real metadata/tag validation
    # is exercised by the evaluation source integration tests.
    monkeypatch.setattr(runner, "validate_source", lambda document, **_kwargs: None)


def _read_short_names(document: Document) -> Document:
    grouped = resolve_short_reporter_colocations(document)
    predicted = predict_short_reporter_fields(grouped, FieldExtractionConfig.from_env())
    return resolve_short_reporter_case_names(predicted)


def _short_stages(create, attribute):
    return (
        (SHORT_PREFIX[0], create),
        (SHORT_PREFIX[1], resolve_short_reporter_colocations),
        (SHORT_PREFIX[2], predict_short_reporter_fields),
        (SHORT_PREFIX[3], resolve_short_reporter_case_names),
        (SHORT_PREFIX[4], attribute),
    )


NONPRIMARY_SETS = (
    "hallucination-set-1",
    "hallucination-set-2",
    "reliable-high-profile",
    "reliable-low-profile",
)


def test_runner_resumes_from_stage_31_without_replaying_discovery_or_calling_models(
    tmp_path: Path, monkeypatch
) -> None:
    text = "Smith v. Jones, 347 U.S. 483 (1954). See Smith, 347 U.S. at 495. Id. at 496."
    source = tmp_path / "filing.txt"
    source.write_text(text, encoding="utf-8")
    document = asyncio.run(grow_roots(Document.from_source(source)))
    document = find_short_reporter_citations(document)
    document = _read_short_names(document)
    document = asyncio.run(attribute_short_reporter_citations(document, review=False))
    document = document.complete_stage("grow_leaves.grow_short_reporter_leaves")
    document = find_reference_citations(document)
    document = asyncio.run(attribute_reference_citations(document, review=False))
    assert document.substage_runs[-1] == "grow_leaves.grow_reference_leaves.attribute_citations"

    input_run = tmp_path / "input-run"
    input_documents = input_run / "documents"
    input_documents.mkdir(parents=True)
    filename = "filing.txt"
    serialized = document.model_dump_json()
    (input_documents / f"{filename}.json").write_text(serialized, encoding="utf-8")
    (input_run / "run.json").write_text(
        json.dumps({"set": "primary", "status": "complete", "filings": [filename]}),
        encoding="utf-8",
    )

    calls = []
    stages = []
    for name, _ in runner._SUBSTAGES:
        if name.endswith("_llm") or name in {
            "grow_leaves.grow_short_reporter_leaves.attribute_citations",
            "grow_leaves.grow_reference_leaves.attribute_citations",
        }:

            async def fake_model_stage(current, substage=name, **_kwargs):
                calls.append(substage)
                return current.complete_substage(substage)

            stages.append((name, fake_model_stage))
        else:

            def fake_rule_stage(current, substage=name, **_kwargs):
                calls.append(substage)
                return current.complete_substage(substage)

            stages.append((name, fake_rule_stage))
    monkeypatch.setattr(runner, "_SUBSTAGES", tuple(stages))
    monkeypatch.setattr(runner, "_RESULTS_ROOT", tmp_path / "runs")

    result = asyncio.run(
        runner.run(
            input_documents,
            input_substage="grow_leaves.grow_reference_leaves.attribute_citations",
            stop_after=runner._SUBSTAGES[-1][0],
        )
    )

    expected = [
        name
        for name, _ in stages
        if name
        not in {
            *SHORT_PREFIX,
            "grow_leaves.grow_short_reporter_leaves.predict_fields",
            "grow_leaves.grow_reference_leaves.find_reference_citations",
            "grow_leaves.grow_reference_leaves.attribute_citations",
        }
    ]
    assert calls == expected
    assert not {
        *SHORT_PREFIX,
        "grow_leaves.grow_reference_leaves.find_reference_citations",
        "grow_leaves.grow_reference_leaves.attribute_citations",
    } & set(calls)
    output = Document.model_validate_json(
        (result / "documents" / f"{filename}.json").read_text(encoding="utf-8")
    )
    assert output.substage_runs[-len(expected) :] == tuple(expected)
    record = json.loads((result / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "complete"
    assert record["input_sha256"][filename] == hashlib.sha256(serialized.encode()).hexdigest()


def _root_input(tmp_path: Path, *, set_name: str = "primary") -> tuple[Path, Document]:
    text = "Smith v. Jones, 347 U.S. 483 (1954). See Smith, 347 U.S. at 495."
    source = tmp_path / "filing.txt"
    source.write_text(text, encoding="utf-8")
    document = asyncio.run(grow_roots(Document.from_source(source)))
    input_documents = tmp_path / "input-run" / "documents"
    input_documents.mkdir(parents=True)
    (input_documents / "filing.txt.json").write_text(document.model_dump_json(), encoding="utf-8")
    (input_documents.parent / "run.json").write_text(
        json.dumps({"set": set_name, "status": "complete", "filings": ["filing.txt"]}), encoding="utf-8"
    )
    return input_documents, document


@pytest.mark.parametrize("set_name", NONPRIMARY_SETS)
def test_runner_infers_nonprimary_set_and_saves_outputs_under_its_corpus(
    tmp_path: Path, monkeypatch, set_name: str
) -> None:
    input_documents, roots = _root_input(tmp_path, set_name=set_name)
    input_path = input_documents / "filing.txt.json"
    original_input = input_path.read_bytes()
    calls = []

    def create(current):
        calls.append("creation")
        return current.complete_substage(
            "grow_leaves.grow_short_reporter_leaves.find_short_reporter_citations"
        )

    async def attribute(current, *, review=True):
        calls.append("attribution")
        assert review is False
        return current.complete_substage("grow_leaves.grow_short_reporter_leaves.attribute_citations")

    monkeypatch.setattr(runner, "attribute_short_reporter_citations", attribute)
    monkeypatch.setattr(
        runner,
        "_SUBSTAGES",
        _short_stages(create, attribute),
    )
    results_root = tmp_path / "runs"
    monkeypatch.setattr(runner, "_RESULTS_ROOT", results_root)

    run_dir = asyncio.run(
        runner.run(
            input_documents,
            input_substage="grow_roots.form_root_groups.form_roots",
            review_leaves=False,
            stop_after=SHORT_PREFIX[-1],
        )
    )

    assert run_dir.parent == results_root / set_name
    assert tuple(results_root.iterdir()) == (results_root / set_name,)
    output = Document.model_validate_json((run_dir / "documents" / "filing.txt.json").read_text())
    assert output.get_stage("grow_roots.form_root_groups") == roots
    assert output.substage_runs[-len(SHORT_PREFIX) :] == SHORT_PREFIX
    assert calls == ["creation", "attribution"]
    record = json.loads((run_dir / "run.json").read_text())
    assert record["set"] == set_name
    assert record["status"] == "complete"
    assert record["input_documents"] == str(input_documents)
    assert record["input_sha256"] == {"filing.txt": hashlib.sha256(original_input).hexdigest()}
    assert input_path.read_bytes() == original_input


@pytest.mark.parametrize(
    ("set_name", "status"),
    [("unknown", "complete"), ("../primary", "complete"), ("primary", "failed"), ("primary", "running")],
)
def test_runner_rejects_unknown_sets_or_incomplete_input_before_writing(
    tmp_path: Path, monkeypatch, set_name: str, status: str
) -> None:
    input_documents, _ = _root_input(tmp_path, set_name=set_name)
    manifest = input_documents.parent / "run.json"
    parent = json.loads(manifest.read_text())
    parent["status"] = status
    manifest.write_text(json.dumps(parent), encoding="utf-8")
    before = manifest.read_bytes()
    results_root = tmp_path / "runs"
    monkeypatch.setattr(runner, "_RESULTS_ROOT", results_root)

    def reject_write(*_args):
        pytest.fail("Invalid input must be rejected before writing")

    monkeypatch.setattr(runner, "_write", reject_write)
    with pytest.raises(ValueError):
        asyncio.run(
            runner.run(
                input_documents,
                input_substage="grow_roots.form_root_groups.form_roots",
                review_leaves=False,
                stop_after=SHORT_PREFIX[-1],
            )
        )

    assert not results_root.exists()
    assert manifest.read_bytes() == before


def test_runner_resumes_the_name_checkpoint_before_short_attribution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_documents, roots = _root_input(tmp_path)
    calls = []
    fail = True

    def create(current):
        calls.append("creation")
        return find_short_reporter_citations(current)

    async def attribute(current, *, review=True, **_kwargs):
        calls.append("attribution")
        assert review is False
        if fail:
            raise RuntimeError("Synthetic interruption before attribution")
        return await attribute_short_reporter_citations(current, review=False)

    monkeypatch.setattr(runner, "attribute_short_reporter_citations", attribute)
    monkeypatch.setattr(
        runner,
        "_SUBSTAGES",
        _short_stages(create, attribute),
    )
    monkeypatch.setattr(runner, "_RESULTS_ROOT", tmp_path / "runs")
    with pytest.raises(RuntimeError, match="Synthetic interruption"):
        asyncio.run(
            runner.run(
                input_documents,
                input_substage="grow_roots.form_root_groups.form_roots",
                review_leaves=False,
                stop_after=SHORT_PREFIX[-1],
            )
        )
    run_dir = next((tmp_path / "runs" / "primary").iterdir())
    saved = Document.model_validate_json((run_dir / "documents" / "filing.txt.json").read_text())
    created = find_short_reporter_citations(roots)
    named = _read_short_names(created)
    assert saved == named
    assert saved.get_substage(SHORT_PREFIX[0]) == created
    assert saved.short_reporters[0].case_name[-1].quote == "Smith"
    assert saved.short_reporters[0].pin_cite[-1].quote == "495"
    assert json.loads((run_dir / "run.json").read_text())["status"] == "failed"

    fail = False
    resumed = asyncio.run(runner.run(None, resume_run=run_dir))
    output = Document.model_validate_json((resumed / "documents" / "filing.txt.json").read_text())
    assert calls == ["creation", "attribution", "attribution"]
    assert output.get_substage(SHORT_PREFIX[0]) == created
    assert output.get_substage(SHORT_PREFIX[3]) == named
    assert output.short_reporters[0].root_id[-1].value == roots.roots[0].id
    assert json.loads((run_dir / "run.json").read_text())["status"] == "complete"


@pytest.mark.parametrize("corruption", ["input", "checkpoint"])
def test_runner_rejects_changed_input_or_an_unexpected_saved_stage(
    tmp_path: Path,
    monkeypatch,
    corruption: str,
) -> None:
    input_documents, roots = _root_input(tmp_path)

    async def interrupt(_current, **_kwargs):
        raise RuntimeError("Synthetic interruption")

    monkeypatch.setattr(runner, "attribute_short_reporter_citations", interrupt)
    monkeypatch.setattr(
        runner,
        "_SUBSTAGES",
        _short_stages(find_short_reporter_citations, interrupt),
    )
    monkeypatch.setattr(runner, "_RESULTS_ROOT", tmp_path / "runs")
    with pytest.raises(RuntimeError):
        asyncio.run(
            runner.run(
                input_documents,
                input_substage="grow_roots.form_root_groups.form_roots",
                review_leaves=False,
                stop_after=SHORT_PREFIX[-1],
            )
        )
    run_dir = next((tmp_path / "runs" / "primary").iterdir())
    if corruption == "input":
        source = input_documents / "filing.txt.json"
        source.write_text(source.read_text() + "\n", encoding="utf-8")
        message = "Leaf input changed"
    else:
        saved = find_short_reporter_citations(roots).complete_substage("unexpected_stage")
        (run_dir / "documents" / "filing.txt.json").write_text(saved.model_dump_json(), encoding="utf-8")
        message = "Unexpected leaf checkpoint"

    with pytest.raises(ValueError, match=message):
        asyncio.run(runner.run(None, resume_run=run_dir))


@pytest.mark.parametrize("corruption", ["saved_set", "parent_set", "parent_status", "parent_filings"])
def test_runner_rejects_resume_corpus_or_parent_manifest_drift_before_writing(
    tmp_path: Path, monkeypatch, corruption: str
) -> None:
    input_documents, _ = _root_input(tmp_path, set_name="hallucination-set-1")
    calls = []

    def create(current):
        calls.append("creation")
        return current.complete_substage(
            "grow_leaves.grow_short_reporter_leaves.find_short_reporter_citations"
        )

    async def interrupt(_current, *, review=True):
        calls.append("attribution")
        assert review is False
        raise RuntimeError("Synthetic interruption")

    monkeypatch.setattr(runner, "attribute_short_reporter_citations", interrupt)
    monkeypatch.setattr(
        runner,
        "_SUBSTAGES",
        _short_stages(create, interrupt),
    )
    results_root = tmp_path / "runs"
    monkeypatch.setattr(runner, "_RESULTS_ROOT", results_root)
    with pytest.raises(RuntimeError, match="Synthetic interruption"):
        asyncio.run(
            runner.run(
                input_documents,
                input_substage="grow_roots.form_root_groups.form_roots",
                review_leaves=False,
                stop_after=SHORT_PREFIX[-1],
            )
        )
    run_dir = next((results_root / "hallucination-set-1").iterdir())
    record_path = run_dir / "run.json"
    parent_path = input_documents.parent / "run.json"
    if corruption == "saved_set":
        record = json.loads(record_path.read_text())
        record["set"] = "unknown"
        record_path.write_text(json.dumps(record), encoding="utf-8")
    else:
        parent = json.loads(parent_path.read_text())
        if corruption == "parent_set":
            parent["set"] = "reliable-low-profile"
        elif corruption == "parent_status":
            parent["status"] = "failed"
        else:
            parent["filings"] = ["other-filing.txt"]
        parent_path.write_text(json.dumps(parent), encoding="utf-8")
    before_record = record_path.read_bytes()
    checkpoint_path = run_dir / "documents" / "filing.txt.json"
    before_checkpoint = checkpoint_path.read_bytes()

    def reject_write(*_args):
        pytest.fail("Invalid resume must be rejected before writing")

    monkeypatch.setattr(runner, "_write", reject_write)
    with pytest.raises(ValueError):
        asyncio.run(runner.run(None, resume_run=run_dir))

    assert calls == ["creation", "attribution"]
    assert record_path.read_bytes() == before_record
    assert checkpoint_path.read_bytes() == before_checkpoint
    assert tuple(results_root.iterdir()) == (results_root / "hallucination-set-1",)


@pytest.mark.parametrize("input_kind", ["root_rule", "default_locator_review"])
def test_runner_restores_upstream_group_after_source_substage_rewind(
    tmp_path: Path, monkeypatch, input_kind: str
) -> None:
    from mellea_lrc.model.execution import get_workflow

    input_documents, source = _root_input(tmp_path)
    if input_kind == "default_locator_review":
        for stage in get_workflow("validate_roots_citation_identity").stages[:3]:
            for substage in stage.substages:
                source = source.complete_substage(substage.name)
            source = source.complete_stage(stage.name)
        upstream_stage = "validate_roots_citation_identity.corroborate_locator_bodies"
        options = {}
    else:
        upstream_stage = "grow_roots.form_root_groups"
        options = {"input_substage": "grow_roots.form_root_groups.form_roots"}
    (input_documents / "filing.txt.json").write_text(source.model_dump_json(), encoding="utf-8")
    expected = source.get_stage(upstream_stage)
    calls = []

    def discover(document: Document) -> Document:
        assert document.get_stage(upstream_stage).model_dump_json() == expected.model_dump_json()
        calls.append(SHORT_PREFIX[0])
        return document.complete_substage(SHORT_PREFIX[0])

    monkeypatch.setattr(runner, "_SUBSTAGES", ((SHORT_PREFIX[0], discover), *runner._SUBSTAGES[1:]))
    monkeypatch.setattr(runner, "_RESULTS_ROOT", tmp_path / "runs")
    run_dir = asyncio.run(
        runner.run(input_documents, review_leaves=False, stop_after=SHORT_PREFIX[0], **options)
    )
    output = Document.model_validate_json((run_dir / "documents" / "filing.txt.json").read_text())
    assert output.get_stage(upstream_stage).model_dump_json() == expected.model_dump_json()
    assert "grow_leaves.grow_short_reporter_leaves" not in output.stage_runs
    assert calls == [SHORT_PREFIX[0]]


def test_partial_supra_rule_checkpoint_runs_enabled_review_before_committing_group(tmp_path, monkeypatch):
    from mellea_lrc.api import grow_leaves

    input_documents, roots = _root_input(tmp_path)
    leaves = asyncio.run(grow_leaves(roots, review_leaves=False))
    rule = "grow_leaves.grow_supra_leaves.attribute_citations_rule"
    original = leaves.get_substage(rule)
    assert "grow_leaves.grow_supra_leaves" not in original.stage_runs
    (input_documents / "filing.txt.json").write_text(original.model_dump_json())
    calls = []

    def fake(substage):
        def finish(document, **_kwargs):
            calls.append(substage)
            if substage == "grow_leaves.grow_supra_leaves.review_attributions_llm":
                assert "grow_leaves.grow_supra_leaves" not in document.stage_runs
            return document.complete_substage(substage)

        return finish

    monkeypatch.setattr(runner, "_SUBSTAGES", tuple((name, fake(name)) for name, _ in runner._SUBSTAGES))
    monkeypatch.setattr(runner, "_RESULTS_ROOT", tmp_path / "results")
    run_dir = asyncio.run(
        runner.run(
            input_documents, input_substage=rule, review_leaves=True, stop_after=runner._SUBSTAGES[-1][0]
        )
    )
    output = Document.model_validate_json((run_dir / "documents" / "filing.txt.json").read_text())
    assert calls == [
        "grow_leaves.grow_supra_leaves.review_attributions_llm",
        "grow_leaves.correct_leaf_readings.correct_leaf_fields",
    ]
    assert output.get_stage("grow_leaves.grow_supra_leaves").substage_runs[-1] == calls[0]
    assert output.get_substage(rule) == original


def test_runner_rejects_missing_earlier_leaf_group_before_execution_or_artifact_write(tmp_path, monkeypatch):
    input_documents, document = _root_input(tmp_path)
    for substage in SHORT_PREFIX:
        document = document.complete_substage(substage)
    last = "grow_leaves.grow_reference_leaves.find_reference_citations"
    document = document.complete_substage(last)
    (input_documents / "filing.txt.json").write_text(document.model_dump_json())

    def unexpected(*_args, **_kwargs):
        pytest.fail("An inconsistent group prefix must not execute any operation")

    monkeypatch.setattr(runner, "_SUBSTAGES", tuple((name, unexpected) for name, _ in runner._SUBSTAGES))
    monkeypatch.setattr(runner, "_RESULTS_ROOT", tmp_path / "results")
    with pytest.raises(ValueError, match=r"missing stage grow_leaves\.grow_short_reporter_leaves"):
        asyncio.run(runner.run(input_documents, input_substage=last))
    assert list((tmp_path / "results").rglob("documents/*.json")) == []


def test_full_runner_uses_workflow_checkpoints_and_resumes_without_granular_dispatch(tmp_path, monkeypatch):
    input_documents, roots = _root_input(tmp_path)
    workflow = runner.grow_leaves
    calls = []
    boundaries = []
    interrupted = False

    async def run_workflow(document, *, review_leaves, checkpoint, field_extraction):
        nonlocal interrupted
        calls.append(document)
        assert review_leaves is False
        assert field_extraction == FieldExtractionConfig(
            ml_enabled=False, case_name="RULE", court="RULE", date="RULE", pin_cite="RULE"
        )

        def save(result):
            nonlocal interrupted
            checkpoint(result)
            boundaries.append((result.runs[-1].kind, result.runs[-1].name))
            if result.runs[-1].name == SHORT_PREFIX[2] and not interrupted:
                interrupted = True
                raise RuntimeError("Interrupted after a persisted workflow checkpoint")

        return await workflow(
            document, review_leaves=review_leaves, checkpoint=save, field_extraction=field_extraction
        )

    def unexpected(*_args, **_kwargs):
        pytest.fail("A full run must delegate to the workflow")

    monkeypatch.setattr(runner, "grow_leaves", run_workflow)
    monkeypatch.setattr(runner, "_SUBSTAGES", tuple((name, unexpected) for name, _ in runner._SUBSTAGES))
    monkeypatch.setattr(runner, "_RESULTS_ROOT", tmp_path / "results")
    with pytest.raises(RuntimeError, match="persisted workflow checkpoint"):
        asyncio.run(
            runner.run(
                input_documents, input_substage="grow_roots.form_root_groups.form_roots", review_leaves=False
            )
        )
    run_dir = next((tmp_path / "results" / "primary").iterdir())
    artifact = run_dir / "documents" / "filing.txt.json"
    saved = Document.model_validate_json(artifact.read_text())
    assert saved.substage_runs[-1] == SHORT_PREFIX[2]
    assert saved.get_stage("grow_roots.form_root_groups") == roots

    assert asyncio.run(runner.run(None, resume_run=run_dir)) == run_dir
    completed = Document.model_validate_json(artifact.read_text())
    expected = tuple(
        name
        for name, _ in runner._SUBSTAGES
        if name != "grow_leaves.grow_supra_leaves.review_attributions_llm"
    )
    assert completed.substage_runs[-len(expected) :] == expected
    assert calls == [roots, saved]
    assert sum(kind == "stage" for kind, _ in boundaries) == 5
    assert [name for kind, name in boundaries if kind == "substage"] == list(expected)
    before = artifact.read_bytes()
    assert asyncio.run(runner.run(None, resume_run=run_dir)) == run_dir
    assert artifact.read_bytes() == before


def test_ml_partial_runner_saves_raw_prediction_and_resumes_without_replaying_inference(
    tmp_path, monkeypatch
):
    input_documents, roots = _root_input(tmp_path)
    original_input = (input_documents / "filing.txt.json").read_bytes()
    settings = {
        "MELLEA_LRC_FIELD_EXTRACTION_ML_ENABLED": "true",
        "MELLEA_LRC_FIELD_EXTRACTION_CASE_NAME": "ML",
        "MELLEA_LRC_FIELD_EXTRACTION_COURT": "RULE",
        "MELLEA_LRC_FIELD_EXTRACTION_DATE": "RULE",
        "MELLEA_LRC_FIELD_EXTRACTION_PIN_CITE": "RULE",
        "MELLEA_LRC_FIELD_EXTRACTION_MODEL_ID": "fixture/bert",
        "MELLEA_LRC_FIELD_EXTRACTION_REVISION": "a" * 40,
        "MELLEA_LRC_FIELD_EXTRACTION_VARIANT": "fixture-augmented",
        "MELLEA_LRC_FIELD_EXTRACTION_DEVICE": "cpu",
        "MELLEA_LRC_FIELD_EXTRACTION_CONTEXT_BEFORE": "260",
        "MELLEA_LRC_FIELD_EXTRACTION_CONTEXT_AFTER": "260",
        "MELLEA_LRC_FIELD_EXTRACTION_BATCH_SIZE": "2",
    }
    config = FieldExtractionConfig.from_env(environ=settings)
    (tmp_path / ".env").write_text("".join(f"{key}={value}\n" for key, value in settings.items()))
    ml_substage = "grow_leaves.grow_short_reporter_leaves.predict_fields"
    calls, inferences = [], []
    interrupted = True

    class Reader:
        def predict_many(self, source, locator_spans, *, contexts):
            inferences.append(locator_spans)
            name_start = source.rindex("Smith", 0, locator_spans[0].start)
            return (
                MLFieldPrediction(
                    model_id=config.model_id,
                    revision=config.revision,
                    variant=config.variant,
                    context_span=Span(0, len(source)),
                    case_name=Span(name_start, name_start + len("Smith")),
                ),
            )

    def predict(document, *, config):
        nonlocal interrupted
        assert config.ml_enabled
        calls.append("ml")
        if interrupted:
            interrupted = False
            raise RuntimeError("Interrupted before span inference")
        return predict_short_reporter_fields(document, config, ml_reader=Reader())

    # The model is supplied by this offline fixture; no checkpoint download is needed.
    monkeypatch.setattr(FieldExtractionConfig, "validate_runtime", lambda self: None)
    monkeypatch.setattr(
        runner,
        "_SUBSTAGES",
        tuple((name, predict if name == ml_substage else call) for name, call in runner._SUBSTAGES),
    )
    monkeypatch.setattr(runner, "_RESULTS_ROOT", tmp_path / "ml-results")
    with pytest.raises(RuntimeError, match="Interrupted before span inference"):
        asyncio.run(
            runner.run(
                input_documents,
                input_substage="grow_roots.form_root_groups.form_roots",
                review_leaves=False,
                stop_after=SHORT_PREFIX[3],
            )
        )
    run_dir = next((tmp_path / "ml-results" / "primary").iterdir())
    artifact = run_dir / "documents" / "filing.txt.json"
    before = Document.model_validate_json(artifact.read_bytes())
    assert before.substage_runs[-1] == SHORT_PREFIX[1]
    assert not inferences

    assert asyncio.run(runner.run(None, resume_run=run_dir)) == run_dir
    output = Document.model_validate_json(artifact.read_bytes())
    assert output.get_substage(SHORT_PREFIX[1]) == before
    assert output.substage_runs[-4:] == SHORT_PREFIX[:4]
    cached = output.get_substage(ml_substage)
    assert not cached.short_reporters[0].case_name
    assert cached.short_reporters[0].ml_field_reading is not None
    assert output.full_locators == roots.full_locators
    assert output.short_reporters[0].case_name[-1].quote == "Smith"
    assert output.short_reporters[0].pin_cite[-1].quote == "495"
    assert len(inferences) == 1
    assert json.loads((run_dir / "run.json").read_text())["field_extraction"] == config.model_dump(
        mode="json"
    )
    assert calls == ["ml", "ml"]
    saved_bytes = artifact.read_bytes()
    assert asyncio.run(runner.run(None, resume_run=run_dir)) == run_dir
    assert artifact.read_bytes() == saved_bytes
    assert calls == ["ml", "ml"]
    assert (input_documents / "filing.txt.json").read_bytes() == original_input


def test_enabled_model_runtime_failure_happens_before_leaf_input_or_artifact_writes(tmp_path, monkeypatch):
    config = FieldExtractionConfig(
        ml_enabled=True,
        case_name="ML",
        court="RULE",
        date="RULE",
        pin_cite="RULE",
        model_id="fixture/bert",
        revision="a" * 40,
        variant="augmented",
        device="cpu",
        context_before=260,
        context_after=260,
        batch_size=2,
    )
    monkeypatch.setattr(FieldExtractionConfig, "from_env", classmethod(lambda cls: config))

    def unavailable(_self):
        raise RuntimeError("Configured checkpoint unavailable")

    monkeypatch.setattr(FieldExtractionConfig, "validate_runtime", unavailable)
    monkeypatch.setattr(runner, "_RESULTS_ROOT", tmp_path / "results")
    with pytest.raises(RuntimeError, match="Configured checkpoint unavailable"):
        asyncio.run(runner.run(tmp_path / "missing-input"))
    assert not (tmp_path / "results").exists()
