"""Pinpoint runs checkpoint each stage, reuse saved inputs, and reject drift."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from evaluations import run_validate_pincite as runner
from mellea_lrc.model import Document, FullReporterCitation, Span
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID
from mellea_lrc.providers.courtlistener import CourtListenerError

STAGES = runner._STAGES


class Client:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def get_opinion(self, opinion_id):
        self.calls.append(opinion_id)
        raise AssertionError("No provider request is needed for these saved-document tests")


def _input(tmp_path: Path, *, filenames=("first.txt", "second.txt"), stages=()):
    input_documents = tmp_path / "input-run" / "documents"
    input_documents.mkdir(parents=True)
    originals = {}
    for name in filenames:
        document = Document.from_source(f"Source for {name}: 550 U.S. 544")
        start = document.text.index("550 U.S. 544")
        citation = FullReporterCitation.from_locator(
            citation_id="reporter-root",
            stage="1_full_reporter_locators",
            source=document.text,
            span=Span(start=start, end=start + len("550 U.S. 544")),
        )
        document = document.add_citation(citation).complete("1_full_reporter_locators")
        citation = citation.record("10_roots").with_root(citation.id)
        document = document.replace_citation(citation).complete("10_roots")
        for stage in stages:
            document = document.complete(stage)
        originals[name] = document
        (input_documents / f"{name}.json").write_text(document.model_dump_json(), encoding="utf-8")
    (input_documents.parent / "run.json").write_text(
        json.dumps({"set": "primary", "status": "complete", "filings": list(filenames)}), encoding="utf-8"
    )
    return input_documents, originals


def _setup(monkeypatch, tmp_path):
    client = Client()
    monkeypatch.setattr(runner, "CourtListenerClient", lambda: client)
    monkeypatch.setattr(runner, "_RESULTS_ROOT", tmp_path / "results")
    return client


def _stage_fakes(monkeypatch, calls, *, failure=None):
    def finish(stage, document):
        calls.append((stage, document.text))
        if failure is not None:
            failure(stage, document)
        return document.complete(stage)

    def retrieve(document, *, client):
        assert isinstance(client, Client)
        return finish(runner.RETRIEVAL_STAGE, document)

    def index(document):
        return finish(runner.INDEX_STAGE, document)

    def resolve(document):
        return finish(runner.RESOLUTION_STAGE, document)

    async def review(document):
        return finish(runner.REVIEW_STAGE, document)

    monkeypatch.setattr(runner, "reporter_root_opinion_retrieval", retrieve)
    monkeypatch.setattr(runner, "index_reporter_root_opinion_pages", index)
    monkeypatch.setattr(runner, "resolve_reporter_citation_pages", resolve)
    monkeypatch.setattr(runner, "review_reporter_citation_opinions", review)

    def prepare(document):
        return finish(runner.EVIDENCE_STAGE, document)

    def judge(document):
        return finish(runner.JUDGMENT_STAGE, document)

    async def propositions(document):
        return finish(runner.PROPOSITION_STAGE, document)

    async def pages(document):
        return finish(runner.PAGE_REVIEW_STAGE, document)

    async def opinions(document):
        return finish(runner.FULL_REVIEW_STAGE, document)

    monkeypatch.setattr(runner, "prepare_reporter_citation_pinpoint_evidence", prepare)
    monkeypatch.setattr(runner, "judge_reporter_citation_pinpoints", judge)
    monkeypatch.setattr(runner, "read_reporter_citation_propositions", propositions)
    monkeypatch.setattr(runner, "review_reporter_citation_pinpoint_pages", pages)
    monkeypatch.setattr(runner, "review_reporter_citation_full_opinions", opinions)


def test_completed_filings_resume_without_stage_or_provider_calls(tmp_path, monkeypatch):
    input_documents, originals = _input(tmp_path)
    client = _setup(monkeypatch, tmp_path)
    calls = []
    _stage_fakes(monkeypatch, calls)
    run_dir = asyncio.run(runner.run(input_documents))
    assert calls == [(stage, original.text) for original in originals.values() for stage in STAGES]
    artifacts = {path.name: path.read_bytes() for path in (run_dir / "documents").iterdir()}
    calls.clear()

    assert asyncio.run(runner.run(None, resume_run=run_dir)) == run_dir

    assert calls == client.calls == []
    assert {path.name: path.read_bytes() for path in (run_dir / "documents").iterdir()} == artifacts
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z", run_dir.name)
    assert run_dir.parent == tmp_path / "results" / "primary"
    record = json.loads((run_dir / "run.json").read_text())
    assert record["workflow"] == "validate_pincite" and record["status"] == "complete"
    assert record["model_profiles"] == {
        stage: asdict(profile) for stage, profile in runner._MODEL_PROFILES.items()
    }
    assert record["stop_after"] == runner.JUDGMENT_STAGE
    for name, original in originals.items():
        path = input_documents / f"{name}.json"
        assert record["input_sha256"][name] == hashlib.sha256(path.read_bytes()).hexdigest()
        saved = Document.model_validate_json(artifacts[f"{name}.json"])
        assert saved.get_stage("10_roots") == original
        assert saved.stage_runs[-len(STAGES) :] == STAGES


def test_resume_rejects_changed_profile_before_rewriting_run_or_documents(tmp_path, monkeypatch):
    input_documents, _originals = _input(tmp_path)
    _setup(monkeypatch, tmp_path)
    calls = []
    _stage_fakes(monkeypatch, calls)
    run_dir = asyncio.run(runner.run(input_documents))
    saved_record = (run_dir / "run.json").read_bytes()
    artifacts = {path.name: path.read_bytes() for path in (run_dir / "documents").iterdir()}
    calls.clear()
    profile = runner._MODEL_PROFILES[runner.PAGE_REVIEW_STAGE]
    monkeypatch.setitem(
        runner._MODEL_PROFILES, runner.PAGE_REVIEW_STAGE, replace(profile, max_tokens=profile.max_tokens + 1)
    )

    with pytest.raises(ValueError, match="model profiles changed"):
        asyncio.run(runner.run(None, resume_run=run_dir))

    assert not calls
    assert (run_dir / "run.json").read_bytes() == saved_record
    assert {path.name: path.read_bytes() for path in (run_dir / "documents").iterdir()} == artifacts


def test_stage_39_input_is_reused_without_retrieval_and_stops_after_41(tmp_path, monkeypatch):
    input_documents, originals = _input(tmp_path, filenames=("first.txt",), stages=(runner.RETRIEVAL_STAGE,))
    client = _setup(monkeypatch, tmp_path)
    calls = []
    _stage_fakes(monkeypatch, calls)
    source_path = input_documents / "first.txt.json"
    original_bytes = source_path.read_bytes()

    def unexpected_retrieval(*_args, **_kwargs):
        pytest.fail("A completed stage 39 input must never be retrieved again")

    monkeypatch.setattr(runner, "reporter_root_opinion_retrieval", unexpected_retrieval)
    run_dir = asyncio.run(runner.run(input_documents, stop_after=runner.RESOLUTION_STAGE))
    output = Document.model_validate_json((run_dir / "documents" / "first.txt.json").read_text())
    original = originals["first.txt"]

    assert calls == [(runner.INDEX_STAGE, original.text), (runner.RESOLUTION_STAGE, original.text)]
    assert client.calls == []
    assert output.get_stage(runner.RETRIEVAL_STAGE) == original
    assert output.stage_runs[-3:] == STAGES[:3]
    assert runner.REVIEW_STAGE not in output.stage_runs
    assert source_path.read_bytes() == original_bytes
    record = json.loads((run_dir / "run.json").read_text())
    assert record["stop_after"] == runner.RESOLUTION_STAGE
    assert record["status"] == "complete"
    assert record["input_sha256"]["first.txt"] == hashlib.sha256(original_bytes).hexdigest()


def test_resume_rejects_changed_input_without_rewriting_completed_artifact(tmp_path, monkeypatch):
    input_documents, _ = _input(tmp_path, filenames=("first.txt",))
    _setup(monkeypatch, tmp_path)
    _stage_fakes(monkeypatch, [])
    run_dir = asyncio.run(runner.run(input_documents))
    artifact = run_dir / "documents" / "first.txt.json"
    saved = artifact.read_bytes()
    source = input_documents / "first.txt.json"
    source.write_text(source.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Pinpoint input changed: first\.txt"):
        asyncio.run(runner.run(None, resume_run=run_dir))

    assert artifact.read_bytes() == saved
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "failed" and "Pinpoint input changed" in record["error"]


@pytest.mark.parametrize("change", ["source", "history", "citation_history"])
def test_resume_rejects_a_saved_document_with_different_input_or_stage_history(tmp_path, monkeypatch, change):
    input_documents, originals = _input(tmp_path, filenames=("first.txt",))
    _setup(monkeypatch, tmp_path)
    calls = []
    _stage_fakes(monkeypatch, calls)
    run_dir = asyncio.run(runner.run(input_documents))
    calls.clear()
    if change == "source":
        corrupt = (
            Document.from_source("Different saved source")
            .complete("1_full_reporter_locators")
            .complete("10_roots")
        )
        for stage in STAGES:
            corrupt = corrupt.complete(stage)
        message = "Saved pinpoint input differs"
    elif change == "history":
        corrupt = originals["first.txt"].complete("different_retrieval_stage")
        message = "Unexpected pinpoint checkpoint"
    else:
        corrupt = originals["first.txt"].get_stage("1_full_reporter_locators")
        citation = corrupt.citations[0].record("10_roots").with_root(WITHDRAWN_ROOT_ID)
        corrupt = corrupt.replace_citation(citation).complete("10_roots")
        for stage in STAGES:
            corrupt = corrupt.complete(stage)
        message = "Saved pinpoint input differs"
    artifact = run_dir / "documents" / "first.txt.json"
    artifact.write_text(corrupt.model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        asyncio.run(runner.run(None, resume_run=run_dir))

    assert calls == []
    assert Document.model_validate_json(artifact.read_text()) == corrupt
    assert json.loads((run_dir / "run.json").read_text())["status"] == "failed"


def test_provider_failure_preserves_finished_documents_and_resumes_only_unfinished_filing(
    tmp_path, monkeypatch
):
    input_documents, originals = _input(tmp_path)
    _setup(monkeypatch, tmp_path)
    inputs_before = {path.name: path.read_bytes() for path in input_documents.iterdir()}
    calls = []
    fail = True

    def failure(stage, document):
        if fail and stage == runner.RETRIEVAL_STAGE and document == originals["second.txt"]:
            raise CourtListenerError("quota exhausted", failure_type="http_error", upstream_status_code=429)

    _stage_fakes(monkeypatch, calls, failure=failure)
    with pytest.raises(CourtListenerError, match="quota exhausted"):
        asyncio.run(runner.run(input_documents))
    run_dir = next((tmp_path / "results" / "primary").iterdir())
    first_path = run_dir / "documents" / "first.txt.json"
    first_saved = first_path.read_bytes()
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "failed"
    assert record["error"] == "CourtListenerError: quota exhausted"
    assert not (run_dir / "documents" / "second.txt.json").exists()
    assert Document.model_validate_json(first_saved).get_stage("10_roots") == originals["first.txt"]
    assert Document.model_validate_json(first_saved).stage_runs[-len(STAGES) :] == STAGES

    fail = False
    calls.clear()
    assert asyncio.run(runner.run(None, resume_run=run_dir)) == run_dir

    assert calls == [(stage, originals["second.txt"].text) for stage in STAGES]
    assert first_path.read_bytes() == first_saved
    assert {path.name: path.read_bytes() for path in input_documents.iterdir()} == inputs_before
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "complete" and "error" not in record
    assert len(tuple((tmp_path / "results" / "primary").iterdir())) == 1


def test_resolution_failure_saves_stage_40_and_resume_runs_only_stage_41(tmp_path, monkeypatch):
    input_documents, originals = _input(tmp_path, filenames=("first.txt",), stages=(runner.RETRIEVAL_STAGE,))
    client = _setup(monkeypatch, tmp_path)
    calls = []
    fail = True

    def failure(stage, _document):
        if fail and stage == runner.RESOLUTION_STAGE:
            raise RuntimeError("Synthetic resolution interruption")

    _stage_fakes(monkeypatch, calls, failure=failure)
    with pytest.raises(RuntimeError, match="Synthetic resolution interruption"):
        asyncio.run(runner.run(input_documents, stop_after=runner.RESOLUTION_STAGE))
    run_dir = next((tmp_path / "results" / "primary").iterdir())
    artifact = run_dir / "documents" / "first.txt.json"
    checkpoint = Document.model_validate_json(artifact.read_text())
    original = originals["first.txt"]
    assert checkpoint == original.complete(runner.INDEX_STAGE)
    assert calls == [(runner.INDEX_STAGE, original.text), (runner.RESOLUTION_STAGE, original.text)]
    assert json.loads((run_dir / "run.json").read_text())["status"] == "failed"

    fail = False
    calls.clear()
    assert asyncio.run(runner.run(None, resume_run=run_dir)) == run_dir

    assert calls == [(runner.RESOLUTION_STAGE, original.text)]
    assert client.calls == []
    output = Document.model_validate_json(artifact.read_text())
    assert output.get_stage(runner.INDEX_STAGE) == checkpoint
    assert output.stage_runs[-1] == runner.RESOLUTION_STAGE
    assert runner.REVIEW_STAGE not in output.stage_runs
    assert json.loads((run_dir / "run.json").read_text())["status"] == "complete"


def test_async_review_failure_preserves_resolution_and_resumes_only_review(tmp_path, monkeypatch):
    input_documents, originals = _input(tmp_path, filenames=("first.txt",), stages=(runner.RETRIEVAL_STAGE,))
    _setup(monkeypatch, tmp_path)
    calls = []
    fail = True

    def failure(stage, _document):
        if fail and stage == runner.REVIEW_STAGE:
            raise RuntimeError("Synthetic opinion review interruption")

    _stage_fakes(monkeypatch, calls, failure=failure)
    with pytest.raises(RuntimeError, match="Synthetic opinion review interruption"):
        asyncio.run(runner.run(input_documents))
    run_dir = next((tmp_path / "results" / "primary").iterdir())
    artifact = run_dir / "documents" / "first.txt.json"
    checkpoint = Document.model_validate_json(artifact.read_text())
    assert checkpoint.stage_runs[-1] == runner.RESOLUTION_STAGE

    fail = False
    calls.clear()
    asyncio.run(runner.run(None, resume_run=run_dir))

    assert calls == [(stage, originals["first.txt"].text) for stage in STAGES[3:]]
    output = Document.model_validate_json(artifact.read_text())
    assert output.get_stage(runner.RESOLUTION_STAGE) == checkpoint
    assert output.stage_runs[-1] == runner.JUDGMENT_STAGE
    assert json.loads((run_dir / "run.json").read_text())["status"] == "complete"


@pytest.mark.parametrize("stages", [(runner.INDEX_STAGE,), (runner.RETRIEVAL_STAGE, runner.RESOLUTION_STAGE)])
def test_nonconsecutive_pinpoint_input_stages_are_rejected_before_stage_calls(tmp_path, monkeypatch, stages):
    input_documents, _ = _input(tmp_path, filenames=("first.txt",), stages=stages)
    _setup(monkeypatch, tmp_path)
    calls = []
    _stage_fakes(monkeypatch, calls)

    with pytest.raises(ValueError, match=r"Unexpected pinpoint input stages: first\.txt"):
        asyncio.run(runner.run(input_documents))

    assert calls == []
    run_dir = next((tmp_path / "results" / "primary").iterdir())
    assert not (run_dir / "documents" / "first.txt.json").exists()
    assert json.loads((run_dir / "run.json").read_text())["status"] == "failed"
