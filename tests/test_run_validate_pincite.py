"""Pinpoint runs checkpoint each substage, reuse saved inputs, and reject drift."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from evaluations import run_validate_pincite as runner
from mellea_lrc.llm.config import LlmOutputMode
from mellea_lrc.llm.profiles import LlmProfile
from mellea_lrc.model import Document, FullReporterCitation, Span
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID
from mellea_lrc.providers.courtlistener import CourtListenerError

SUBSTAGES = runner._SUBSTAGES


@pytest.fixture(autouse=True)
def configured_profiles(monkeypatch):
    profiles = {
        substage: LlmProfile(
            name=f"test_profile_{index}",
            model=f"test_model_{index}",
            api_base="https://example.invalid/v1",
            api_key_env="TEST_API_KEY",
            temperature=0.0,
            timeout_seconds=30.0,
            max_tokens=1000,
            max_attempts=2,
            output_mode=LlmOutputMode.JSON_SCHEMA,
        )
        for index, substage in enumerate(runner._MODEL_SUBSTAGES)
    }
    monkeypatch.setattr(runner, "load_profile", lambda substage: profiles[substage])
    return profiles


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
            substage="grow_roots.locator_discovery.full_reporter_locators",
            source=document.text,
            span=Span(start=start, end=start + len("550 U.S. 544")),
        )
        document = document.add_citation(citation).complete_substage(
            "grow_roots.locator_discovery.full_reporter_locators"
        )
        citation = citation.record("grow_roots.root_formation.rule").with_root(citation.id)
        document = document.replace_citation(citation).complete_substage("grow_roots.root_formation.rule")
        for substage in stages:
            document = document.complete_substage(substage)
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
    def finish(substage, document):
        calls.append((substage, document.text))
        if failure is not None:
            failure(substage, document)
        return document.complete_substage(substage)

    def retrieve(document, *, client):
        assert isinstance(client, Client)
        return finish(runner.RETRIEVAL_SUBSTAGE, document)

    def index(document):
        return finish(runner.INDEX_SUBSTAGE, document)

    def resolve(document):
        return finish(runner.RESOLUTION_SUBSTAGE, document)

    async def review(document):
        return finish(runner.REVIEW_SUBSTAGE, document)

    monkeypatch.setattr(runner, "reporter_root_opinion_retrieval", retrieve)
    monkeypatch.setattr(runner, "index_reporter_root_opinion_pages", index)
    monkeypatch.setattr(runner, "resolve_reporter_citation_pages", resolve)
    monkeypatch.setattr(runner, "review_reporter_citation_opinions", review)

    def prepare(document):
        return finish(runner.EVIDENCE_SUBSTAGE, document)

    def judge(document):
        return finish(runner.JUDGMENT_SUBSTAGE, document)

    async def propositions(document):
        return finish(runner.PROPOSITION_SUBSTAGE, document)

    async def pages(document):
        return finish(runner.PAGE_REVIEW_SUBSTAGE, document)

    async def opinions(document):
        return finish(runner.FULL_REVIEW_SUBSTAGE, document)

    monkeypatch.setattr(runner, "prepare_reporter_citation_pinpoint_evidence", prepare)
    monkeypatch.setattr(runner, "judge_reporter_citation_pinpoints", judge)
    monkeypatch.setattr(runner, "read_reporter_citation_propositions", propositions)
    monkeypatch.setattr(runner, "review_reporter_citation_pinpoint_pages", pages)
    monkeypatch.setattr(runner, "review_reporter_citation_full_opinions", opinions)


def test_completed_filings_resume_without_stage_or_provider_calls(tmp_path, monkeypatch, configured_profiles):
    input_documents, originals = _input(tmp_path)
    client = _setup(monkeypatch, tmp_path)
    calls = []
    _stage_fakes(monkeypatch, calls)
    run_dir = asyncio.run(runner.run(input_documents))
    assert calls == [(substage, original.text) for original in originals.values() for substage in SUBSTAGES]
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
        substage: asdict(profile) for substage, profile in configured_profiles.items()
    }
    assert record["stop_after"] == runner.JUDGMENT_SUBSTAGE
    for name, original in originals.items():
        path = input_documents / f"{name}.json"
        assert record["input_sha256"][name] == hashlib.sha256(path.read_bytes()).hexdigest()
        saved = Document.model_validate_json(artifacts[f"{name}.json"])
        assert saved.get_substage("grow_roots.root_formation.rule") == original
        assert saved.substage_runs[-len(SUBSTAGES) :] == SUBSTAGES


def test_resume_rejects_changed_profile_before_rewriting_run_or_documents(
    tmp_path, monkeypatch, configured_profiles
):
    input_documents, _originals = _input(tmp_path)
    _setup(monkeypatch, tmp_path)
    calls = []
    _stage_fakes(monkeypatch, calls)
    run_dir = asyncio.run(runner.run(input_documents))
    saved_record = (run_dir / "run.json").read_bytes()
    artifacts = {path.name: path.read_bytes() for path in (run_dir / "documents").iterdir()}
    calls.clear()
    profile = configured_profiles[runner.PAGE_REVIEW_SUBSTAGE]
    monkeypatch.setitem(
        configured_profiles,
        runner.PAGE_REVIEW_SUBSTAGE,
        replace(profile, max_tokens=profile.max_tokens + 1),
    )

    with pytest.raises(ValueError, match="model profiles changed"):
        asyncio.run(runner.run(None, resume_run=run_dir))

    assert not calls
    assert (run_dir / "run.json").read_bytes() == saved_record
    assert {path.name: path.read_bytes() for path in (run_dir / "documents").iterdir()} == artifacts


def test_stage_39_input_is_reused_without_retrieval_and_stops_after_41(
    tmp_path, monkeypatch, configured_profiles
):
    input_documents, originals = _input(
        tmp_path, filenames=("first.txt",), stages=(runner.RETRIEVAL_SUBSTAGE,)
    )
    client = _setup(monkeypatch, tmp_path)
    calls = []
    _stage_fakes(monkeypatch, calls)
    source_path = input_documents / "first.txt.json"
    original_bytes = source_path.read_bytes()

    def unexpected_retrieval(*_args, **_kwargs):
        pytest.fail("A completed substage 39 input must never be retrieved again")

    monkeypatch.setattr(runner, "reporter_root_opinion_retrieval", unexpected_retrieval)
    run_dir = asyncio.run(runner.run(input_documents, stop_after=runner.RESOLUTION_SUBSTAGE))
    output = Document.model_validate_json((run_dir / "documents" / "first.txt.json").read_text())
    original = originals["first.txt"]

    assert calls == [(runner.INDEX_SUBSTAGE, original.text), (runner.RESOLUTION_SUBSTAGE, original.text)]
    assert client.calls == []
    assert output.get_substage(runner.RETRIEVAL_SUBSTAGE) == original
    assert output.substage_runs[-3:] == SUBSTAGES[:3]
    assert runner.REVIEW_SUBSTAGE not in output.substage_runs
    assert source_path.read_bytes() == original_bytes
    record = json.loads((run_dir / "run.json").read_text())
    assert record["stop_after"] == runner.RESOLUTION_SUBSTAGE
    assert record["status"] == "complete"
    assert record["model_profiles"] == {
        substage: asdict(profile) for substage, profile in configured_profiles.items()
    }
    assert record["input_sha256"]["first.txt"] == hashlib.sha256(original_bytes).hexdigest()


def test_profiles_resolve_once_at_run_start_without_credentials(tmp_path, monkeypatch, configured_profiles):
    input_documents, _ = _input(tmp_path, filenames=("first.txt",))
    _setup(monkeypatch, tmp_path)
    calls = []
    _stage_fakes(monkeypatch, calls)
    loaded = []

    def load_profile(substage):
        loaded.append(substage)
        return configured_profiles[substage]

    def unexpected_credentials(*_args, **_kwargs):
        pytest.fail("Recording profile descriptors must not resolve credentials")

    monkeypatch.setattr(runner, "load_profile", load_profile)
    monkeypatch.setattr(LlmProfile, "resolve", unexpected_credentials)
    run_dir = asyncio.run(runner.run(input_documents, stop_after=runner.RETRIEVAL_SUBSTAGE))
    assert loaded == list(runner._MODEL_SUBSTAGES)
    assert calls == [(runner.RETRIEVAL_SUBSTAGE, "Source for first.txt: 550 U.S. 544")]
    record = json.loads((run_dir / "run.json").read_text())
    assert set(record["model_profiles"]) == set(runner._MODEL_SUBSTAGES)
    assert all("api_key" not in profile for profile in record["model_profiles"].values())


def test_missing_profile_binding_fails_before_writes_or_client_creation(
    tmp_path, monkeypatch, configured_profiles
):
    input_documents, _ = _input(tmp_path, filenames=("first.txt",))
    _setup(monkeypatch, tmp_path)
    loaded = []

    def load_profile(substage):
        loaded.append(substage)
        if substage == runner.FULL_REVIEW_SUBSTAGE:
            raise RuntimeError("Missing environment profile binding")
        return configured_profiles[substage]

    def unexpected_client():
        pytest.fail("Missing configuration must fail before opening a provider client")

    monkeypatch.setattr(runner, "load_profile", load_profile)
    monkeypatch.setattr(runner, "CourtListenerClient", unexpected_client)
    with pytest.raises(RuntimeError, match="Missing environment profile binding"):
        asyncio.run(runner.run(input_documents, stop_after=runner.RETRIEVAL_SUBSTAGE))
    assert loaded == list(runner._MODEL_SUBSTAGES)
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("timing", ["before", "during"])
def test_profile_drift_cannot_persist_model_stage_and_resumes_after_restore(
    tmp_path, monkeypatch, configured_profiles, timing
):
    input_documents, originals = _input(tmp_path, filenames=("first.txt",))
    client = _setup(monkeypatch, tmp_path)
    calls = []
    original_profile = configured_profiles[runner.REVIEW_SUBSTAGE]
    change_at = runner.RESOLUTION_SUBSTAGE if timing == "before" else runner.REVIEW_SUBSTAGE
    drift = True

    def change_configuration(substage, _document):
        if drift and substage == change_at:
            configured_profiles[runner.REVIEW_SUBSTAGE] = replace(
                original_profile, max_tokens=original_profile.max_tokens + 1
            )

    _stage_fakes(monkeypatch, calls, failure=change_configuration)
    with pytest.raises(ValueError, match="model profiles changed during execution"):
        asyncio.run(runner.run(input_documents))

    run_dir = next((tmp_path / "results" / "primary").iterdir())
    artifact = run_dir / "documents" / "first.txt.json"
    checkpoint = Document.model_validate_json(artifact.read_text())
    assert checkpoint.substage_runs[-1] == runner.RESOLUTION_SUBSTAGE
    assert runner.REVIEW_SUBSTAGE not in checkpoint.substage_runs
    assert calls == [
        (substage, originals["first.txt"].text) for substage in SUBSTAGES[: 3 if timing == "before" else 4]
    ]
    assert client.calls == []
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "failed"
    assert record["model_profiles"][runner.REVIEW_SUBSTAGE] == asdict(original_profile)

    configured_profiles[runner.REVIEW_SUBSTAGE] = original_profile
    drift = False
    calls.clear()
    assert asyncio.run(runner.run(None, resume_run=run_dir)) == run_dir
    assert calls == [(substage, originals["first.txt"].text) for substage in SUBSTAGES[3:]]
    completed = Document.model_validate_json(artifact.read_text())
    assert completed.get_substage(runner.RESOLUTION_SUBSTAGE) == checkpoint
    assert json.loads((run_dir / "run.json").read_text())["status"] == "complete"


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
            .complete_substage("grow_roots.locator_discovery.full_reporter_locators")
            .complete_substage("grow_roots.root_formation.rule")
        )
        for substage in SUBSTAGES:
            corrupt = corrupt.complete_substage(substage)
        message = "Saved pinpoint input differs"
    elif change == "history":
        corrupt = originals["first.txt"].complete_substage("different_retrieval_stage")
        message = "Unexpected pinpoint checkpoint"
    else:
        corrupt = originals["first.txt"].get_substage("grow_roots.locator_discovery.full_reporter_locators")
        citation = corrupt.citations[0].record("grow_roots.root_formation.rule").with_root(WITHDRAWN_ROOT_ID)
        corrupt = corrupt.replace_citation(citation).complete_substage("grow_roots.root_formation.rule")
        for substage in SUBSTAGES:
            corrupt = corrupt.complete_substage(substage)
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

    def failure(substage, document):
        if fail and substage == runner.RETRIEVAL_SUBSTAGE and document == originals["second.txt"]:
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
    assert (
        Document.model_validate_json(first_saved).get_substage("grow_roots.root_formation.rule")
        == originals["first.txt"]
    )
    assert Document.model_validate_json(first_saved).substage_runs[-len(SUBSTAGES) :] == SUBSTAGES

    fail = False
    calls.clear()
    assert asyncio.run(runner.run(None, resume_run=run_dir)) == run_dir

    assert calls == [(substage, originals["second.txt"].text) for substage in SUBSTAGES]
    assert first_path.read_bytes() == first_saved
    assert {path.name: path.read_bytes() for path in input_documents.iterdir()} == inputs_before
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "complete" and "error" not in record
    assert len(tuple((tmp_path / "results" / "primary").iterdir())) == 1


def test_resolution_failure_saves_stage_40_and_resume_runs_only_stage_41(tmp_path, monkeypatch):
    input_documents, originals = _input(
        tmp_path, filenames=("first.txt",), stages=(runner.RETRIEVAL_SUBSTAGE,)
    )
    client = _setup(monkeypatch, tmp_path)
    calls = []
    fail = True

    def failure(substage, _document):
        if fail and substage == runner.RESOLUTION_SUBSTAGE:
            raise RuntimeError("Synthetic resolution interruption")

    _stage_fakes(monkeypatch, calls, failure=failure)
    with pytest.raises(RuntimeError, match="Synthetic resolution interruption"):
        asyncio.run(runner.run(input_documents, stop_after=runner.RESOLUTION_SUBSTAGE))
    run_dir = next((tmp_path / "results" / "primary").iterdir())
    artifact = run_dir / "documents" / "first.txt.json"
    checkpoint = Document.model_validate_json(artifact.read_text())
    original = originals["first.txt"]
    assert checkpoint == original.complete_substage(runner.INDEX_SUBSTAGE).complete_stage(
        "validate_pincite.opinion_preparation"
    )
    assert calls == [(runner.INDEX_SUBSTAGE, original.text), (runner.RESOLUTION_SUBSTAGE, original.text)]
    assert json.loads((run_dir / "run.json").read_text())["status"] == "failed"

    fail = False
    calls.clear()
    assert asyncio.run(runner.run(None, resume_run=run_dir)) == run_dir

    assert calls == [(runner.RESOLUTION_SUBSTAGE, original.text)]
    assert client.calls == []
    output = Document.model_validate_json(artifact.read_text())
    assert output.get_stage("validate_pincite.opinion_preparation") == checkpoint
    assert output.get_substage(runner.INDEX_SUBSTAGE) == checkpoint.get_substage(runner.INDEX_SUBSTAGE)
    assert output.substage_runs[-1] == runner.RESOLUTION_SUBSTAGE
    assert runner.REVIEW_SUBSTAGE not in output.substage_runs
    assert json.loads((run_dir / "run.json").read_text())["status"] == "complete"


def test_async_review_failure_preserves_resolution_and_resumes_only_review(tmp_path, monkeypatch):
    input_documents, originals = _input(
        tmp_path, filenames=("first.txt",), stages=(runner.RETRIEVAL_SUBSTAGE,)
    )
    _setup(monkeypatch, tmp_path)
    calls = []
    fail = True

    def failure(substage, _document):
        if fail and substage == runner.REVIEW_SUBSTAGE:
            raise RuntimeError("Synthetic opinion review interruption")

    _stage_fakes(monkeypatch, calls, failure=failure)
    with pytest.raises(RuntimeError, match="Synthetic opinion review interruption"):
        asyncio.run(runner.run(input_documents))
    run_dir = next((tmp_path / "results" / "primary").iterdir())
    artifact = run_dir / "documents" / "first.txt.json"
    checkpoint = Document.model_validate_json(artifact.read_text())
    assert checkpoint.substage_runs[-1] == runner.RESOLUTION_SUBSTAGE

    fail = False
    calls.clear()
    asyncio.run(runner.run(None, resume_run=run_dir))

    assert calls == [(substage, originals["first.txt"].text) for substage in SUBSTAGES[3:]]
    output = Document.model_validate_json(artifact.read_text())
    assert output.get_substage(runner.RESOLUTION_SUBSTAGE) == checkpoint
    assert output.substage_runs[-1] == runner.JUDGMENT_SUBSTAGE
    assert json.loads((run_dir / "run.json").read_text())["status"] == "complete"


@pytest.mark.parametrize(
    "stages", [(runner.INDEX_SUBSTAGE,), (runner.RETRIEVAL_SUBSTAGE, runner.RESOLUTION_SUBSTAGE)]
)
def test_nonconsecutive_pinpoint_input_substages_are_rejected_before_stage_calls(
    tmp_path, monkeypatch, stages
):
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


def test_runner_rejects_missing_earlier_group_before_provider_work_or_artifact_write(tmp_path, monkeypatch):
    input_documents, _ = _input(tmp_path, filenames=("first.txt",), stages=SUBSTAGES[:3])
    client = _setup(monkeypatch, tmp_path)
    calls = []
    _stage_fakes(monkeypatch, calls)
    with pytest.raises(ValueError, match=r"missing stage validate_pincite\.opinion_preparation"):
        asyncio.run(runner.run(input_documents))
    assert calls == client.calls == []
    assert list((tmp_path / "results").rglob("documents/*.json")) == []
