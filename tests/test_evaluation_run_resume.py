"""Offline checks for resuming one durable evaluation run."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path

import httpx
import pytest

from evaluations import __main__ as runner
from evaluations import complete_stage_boundary
from mellea_lrc import api
from mellea_lrc.api import Document, grow_roots
from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.model import (
    DocketLookup,
    DocketLookupAttempt,
    DocketLookupFailure,
    FullDocketCitation,
    Span,
)
from mellea_lrc.model.citations.body_evidence import (
    BodyEvidenceFailure,
    BodySearch,
    BodySearchAttempt,
    BodySource,
)
from mellea_lrc.model.citations.field_body_evidence import FieldBodySearch
from mellea_lrc.model.citations.govinfo_lookup import GovInfoDocketLookup, GovInfoLookupAttempt
from mellea_lrc.model.execution import stage_for_substage
from mellea_lrc.providers.courtlistener import CourtListenerClient

_RULE_FIELDS = FieldExtractionConfig(
    ml_enabled=False,
    case_name="RULE",
    court="RULE",
    date="RULE",
    pin_cite="RULE",
    context_before=260,
    context_after=260,
)

_ROOT_WORKFLOW = importlib.import_module("mellea_lrc.workflows.validate_roots_citation_identity")
_LOCATOR_REVIEW = _ROOT_WORKFLOW.corroborate_locator_bodies
_BODY_START = runner._RUN_SUBSTAGES.index(runner._COURTLISTENER_OPINION_SUBSTAGE)
_BODY_INPUT_SUBSTAGES = runner._RUN_SUBSTAGES[:_BODY_START]
_BODY_SUBSTAGES = runner._RUN_SUBSTAGES[_BODY_START:]


def _ml_field_settings() -> dict[str, str]:
    return {
        "MELLEA_LRC_FIELD_EXTRACTION_ML_ENABLED": "true",
        "MELLEA_LRC_FIELD_EXTRACTION_CASE_NAME": "ML",
        "MELLEA_LRC_FIELD_EXTRACTION_COURT": "ML",
        "MELLEA_LRC_FIELD_EXTRACTION_DATE": "ML",
        "MELLEA_LRC_FIELD_EXTRACTION_PIN_CITE": "RULE",
        "MELLEA_LRC_FIELD_EXTRACTION_MODEL_ID": "fixture/bert",
        "MELLEA_LRC_FIELD_EXTRACTION_REVISION": "a" * 40,
        "MELLEA_LRC_FIELD_EXTRACTION_VARIANT": "fixture-augmented",
        "MELLEA_LRC_FIELD_EXTRACTION_DEVICE": "cpu",
        "MELLEA_LRC_FIELD_EXTRACTION_CONTEXT_BEFORE": "260",
        "MELLEA_LRC_FIELD_EXTRACTION_CONTEXT_AFTER": "260",
        "MELLEA_LRC_FIELD_EXTRACTION_BATCH_SIZE": "2",
    }


def test_end_to_end_requires_shared_prediction_before_the_same_consuming_field_steps():
    root = "grow_roots.read_root_fields.predict_fields"
    short = "grow_leaves.grow_short_reporter_leaves.predict_fields"
    selected = runner._E2E_SUBSTAGES
    assert selected.count(root) == selected.count(short) == 1
    assert root in runner._ROOT_SUBSTAGES
    assert selected.index(root) == selected.index("grow_roots.read_root_fields.resolve_colocations") + 1
    assert (
        selected.index(short)
        == selected.index("grow_leaves.grow_short_reporter_leaves.resolve_colocations") + 1
    )
    assert selected[selected.index(root) + 1].endswith(".resolve_case_names")
    assert selected[selected.index(short) + 1].endswith(".resolve_case_names")


@pytest.mark.parametrize("change", ["missing", "field_source", "revision", "context"])
def test_resume_rejects_field_configuration_drift_before_work_or_writes(tmp_path, monkeypatch, change):
    settings = _ml_field_settings()
    saved_config = FieldExtractionConfig.from_env(environ=settings)
    run_dir = tmp_path / "saved-run"
    artifacts = run_dir / "documents"
    artifacts.mkdir(parents=True)
    artifact = artifacts / "001.txt.json"
    artifact.write_text(Document.from_source("Original source.").model_dump_json())
    record = {"set": "primary", "status": "failed", "field_extraction": saved_config.model_dump(mode="json")}
    if change == "missing":
        record.pop("field_extraction")
    elif change == "field_source":
        settings["MELLEA_LRC_FIELD_EXTRACTION_CASE_NAME"] = "RULE"
    elif change == "revision":
        settings["MELLEA_LRC_FIELD_EXTRACTION_REVISION"] = "b" * 40
    else:
        settings["MELLEA_LRC_FIELD_EXTRACTION_CONTEXT_BEFORE"] = "128"
    (tmp_path / ".env").write_text("".join(f"{key}={value}\n" for key, value in settings.items()))
    record_path = run_dir / "run.json"
    record_path.write_text(json.dumps(record))
    before_record, before_artifact = record_path.read_bytes(), artifact.read_bytes()

    def unexpected(*_args, **_kwargs):
        pytest.fail(
            "Field configuration drift must fail before loading sources, calling providers, or writing"
        )

    for name in (
        "grow_roots",
        "load_source",
        "CourtListenerClient",
        "_configured_model_profiles",
        "_write_json",
    ):
        monkeypatch.setattr(runner, name, unexpected)
    with pytest.raises(ValueError, match="Field extraction settings differ from saved provenance"):
        asyncio.run(runner._run(tmp_path / "unused", tmp_path / "unused", None, resume_run=run_dir))
    assert record_path.read_bytes() == before_record
    assert artifact.read_bytes() == before_artifact


def _dataset(tmp_path: Path, filenames: tuple[str, ...]) -> Path:
    data_root = tmp_path / "data"
    source_dir = data_root / "primary" / "documents_txt"
    source_dir.mkdir(parents=True)
    for filename in filenames:
        (source_dir / filename).write_text(f"source for {filename}\n", encoding="utf-8")
    (data_root / "primary" / "documents.json").write_text(
        json.dumps({"documents": {name: {} for name in filenames}}), encoding="utf-8"
    )
    return data_root


def _courtlistener_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "MELLEA_LRC_FIELD_EXTRACTION_ML_ENABLED=false\nMELLEA_LRC_FIELD_EXTRACTION_CASE_NAME=RULE\nMELLEA_LRC_FIELD_EXTRACTION_COURT=RULE\nMELLEA_LRC_FIELD_EXTRACTION_DATE=RULE\nMELLEA_LRC_FIELD_EXTRACTION_PIN_CITE=RULE\nMELLEA_LRC_FIELD_EXTRACTION_CONTEXT_BEFORE=260\nMELLEA_LRC_FIELD_EXTRACTION_CONTEXT_AFTER=260\n"
        "COURTLISTENER_BASE_URL=https://proxy.example/api/rest/v4/\n"
        "COURTLISTENER_TIMEOUT_SECONDS=47\n"
        "COURTLISTENER_API_TOKEN=unused-main-token\n"
        "COURTLISTENER_API_TOKEN_RESERVED=offline-reserved-token\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COURTLISTENER_BASE_URL", "https://process.example/ignored/")
    monkeypatch.setenv("COURTLISTENER_TIMEOUT_SECONDS", "999")
    monkeypatch.setenv("COURTLISTENER_API_TOKEN_RESERVED", "ignored-process-token")
    return env_path


def _annotation_cutoffs(data_root: Path, dates: dict[str, str], groups: dict[str, str] | None = None) -> None:
    documents = data_root / "primary" / "documents"
    documents.mkdir(exist_ok=True)
    groups = groups or {name: name for name in dates}
    for filename, filed_on in dates.items():
        source = groups[filename]
        header = {
            "unit": "header",
            "document": filename,
            "filing": {
                "date": filed_on,
                "provenance": {
                    "source_pdf": filename.removesuffix(".txt") + ".pdf",
                    "page": 1,
                    "basis": "ecf_header",
                    "evidence": "Filed",
                },
            },
            "case_cutoff": {"date": dates[source], "source_document": source},
        }
        (documents / f"{Path(filename).stem}.jsonl").write_text(json.dumps(header) + "\n", encoding="utf-8")


def _patch_operation(monkeypatch: pytest.MonkeyPatch, name: str, value: object) -> None:
    operation = importlib.import_module(getattr(api, name).__module__)
    stage = stage_for_substage(operation.SUBSTAGE)
    assert stage is not None, f"Unknown workflow operation: {name}"
    module = importlib.import_module(f"mellea_lrc.workflows.{stage}")
    monkeypatch.setattr(module, name, value)


def _complete(document: Document, stages: tuple[str, ...]) -> Document:
    for substage in stages:
        document = document.complete_substage(substage)
        document = complete_stage_boundary(document, runner._FIELD_RUN_SUBSTAGES)
    return document


def _complete_with_checkpoints(
    document: Document,
    stages: tuple[str, ...],
    checkpoint: Callable[[Document], None] | None,
) -> Document:
    for substage in stages:
        if substage in document.substage_runs:
            continue
        document = document.complete_substage(substage)
        if checkpoint is not None:
            checkpoint(document)
        grouped = complete_stage_boundary(document, runner._FIELD_RUN_SUBSTAGES)
        if grouped != document and checkpoint is not None:
            checkpoint(grouped)
        document = grouped
    return document


@pytest.fixture(autouse=True)
def _offline_body_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "MELLEA_LRC_FIELD_EXTRACTION_ML_ENABLED=false\nMELLEA_LRC_FIELD_EXTRACTION_CASE_NAME=RULE\nMELLEA_LRC_FIELD_EXTRACTION_COURT=RULE\nMELLEA_LRC_FIELD_EXTRACTION_DATE=RULE\nMELLEA_LRC_FIELD_EXTRACTION_PIN_CITE=RULE\nMELLEA_LRC_FIELD_EXTRACTION_CONTEXT_BEFORE=260\nMELLEA_LRC_FIELD_EXTRACTION_CONTEXT_AFTER=260\n"
    )
    monkeypatch.chdir(tmp_path)
    # These tests exercise checkpoint orchestration; native source metadata has
    # dedicated coverage in test_evaluation_end_to_end.py.
    monkeypatch.setattr(runner, "load_source", lambda path: Document.from_source(path))

    def unexpected_network(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Offline evaluation tests must not call a real provider")

    monkeypatch.setattr(httpx.Client, "request", unexpected_network)
    monkeypatch.setattr(httpx.AsyncClient, "request", unexpected_network)

    async def body(
        document: Document,
        *,
        retrospective_date: date | None = None,
        courtlistener_client: object | None = None,
        checkpoint: Callable[[Document], None] | None = None,
    ) -> Document:
        assert document.substage_runs[:_BODY_START] == _BODY_INPUT_SUBSTAGES
        return _complete_with_checkpoints(document, _BODY_SUBSTAGES, checkpoint)

    monkeypatch.setattr(_ROOT_WORKFLOW, "corroborate_locator_bodies", body)


def test_resume_skips_valid_documents_and_reuses_timestamp_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filenames = ("001.txt", "002.txt")
    data_root = _dataset(tmp_path, filenames)
    calls: list[str] = []

    async def fake_grow(
        document: Document,
        *,
        hunt_dockets: bool,
        review_docket_roots: bool,
        field_extraction,
        checkpoint=None,
    ) -> Document:
        assert hunt_dockets and review_docket_roots
        calls.append(Path(document.source_path or "").name)
        return _complete(document, runner._RUN_SUBSTAGES[:12])

    async def fake_validate(
        document: Document,
        *,
        retrospective_date: date | None = None,
        checkpoint: Callable[[Document], None] | None = None,
    ) -> Document:
        document = _complete_with_checkpoints(document, runner._RUN_SUBSTAGES[12:_BODY_START], checkpoint)
        return _complete_with_checkpoints(document, _BODY_SUBSTAGES, checkpoint)

    monkeypatch.setattr(runner, "grow_roots", fake_grow)
    monkeypatch.setattr(runner, "validate_roots_citation_identity", fake_validate)
    run_dir = asyncio.run(runner._run(data_root, tmp_path / "results", None))
    assert calls == list(filenames)

    (run_dir / "documents" / "002.txt.json").unlink()
    record_path = run_dir / "run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["status"] = "failed"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    assert asyncio.run(runner._run(tmp_path / "ignored", tmp_path / "unused", None, run_dir)) == run_dir
    assert calls == [*filenames, "002.txt"]
    assert not (run_dir / "checkpoints").exists()
    assert json.loads(record_path.read_text(encoding="utf-8"))["status"] == "complete"

    (data_root / "primary" / "documents_txt" / "002.txt").write_text("changed source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Run source content differs"):
        asyncio.run(runner._run(tmp_path / "ignored", tmp_path / "unused", None, run_dir))
    assert calls == [*filenames, "002.txt"]


@pytest.mark.parametrize(
    "current_name, saved_name",
    [
        ("validate_roots_citation_identity", "validate_roots"),
        ("validate_all_citation_content", "validate_pincite"),
    ],
)
def test_resume_rejects_old_workflow_manifest_names_before_calls_or_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, current_name: str, saved_name: str
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    document = Document.from_source(data_root / "primary" / "documents_txt" / "001.txt").complete_substage(
        "grow_roots.discover_root_locators.find_full_reporter_locators"
    )
    assert document.stage_runs == ()
    run_dir = tmp_path / "saved-run"
    artifacts = run_dir / "documents"
    artifacts.mkdir(parents=True)
    artifact = artifacts / "001.txt.json"
    artifact.write_text(document.model_dump_json(), encoding="utf-8")
    # A valid current atomic checkpoint does not make an obsolete workflow
    # manifest resumable.
    assert Document.model_validate_json(artifact.read_bytes()) == document
    record_path = run_dir / "run.json"
    record_path.write_text(
        json.dumps(
            {
                "set": "primary",
                "status": "failed",
                "filings": ["001.txt"],
                "data_root": str(data_root),
                "end_to_end": True,
                "workflows": [saved_name if name == current_name else name for name in runner._E2E_WORKFLOWS],
                "workers": 3,
                "model_profiles": {},
                "field_extraction": _RULE_FIELDS.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    saved_record = record_path.read_bytes()
    saved_artifact = artifact.read_bytes()

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Old workflow names must fail before source, profile, provider, or workflow calls")

    for name in (*runner._E2E_WORKFLOWS, "CourtListenerClient", "load_source", "_configured_model_profiles"):
        monkeypatch.setattr(runner, name, unexpected)
    with pytest.raises(ValueError, match="Saved end-to-end workflow names differ"):
        asyncio.run(runner._run(tmp_path / "unused", tmp_path / "unused", None, resume_run=run_dir))
    assert record_path.read_bytes() == saved_record
    assert artifact.read_bytes() == saved_artifact


def test_resume_rejects_a_cumulative_document_with_a_stage_gap_before_provider_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    results_root = tmp_path / "results"

    async def fake_grow(document: Document, **_kwargs: object) -> Document:
        return _complete(document, runner._RUN_SUBSTAGES[:12])

    async def interrupt_validation(document: Document, **_kwargs: object) -> Document:
        raise RuntimeError("interrupted after growth")

    monkeypatch.setattr(runner, "grow_roots", fake_grow)
    monkeypatch.setattr(runner, "validate_roots_citation_identity", interrupt_validation)
    with pytest.raises(RuntimeError, match="interrupted after growth"):
        asyncio.run(runner._run(data_root, results_root, None))
    run_dir = next(results_root.iterdir())
    artifact = run_dir / "documents" / "001.txt.json"
    interrupted = Document.model_validate_json(artifact.read_text(encoding="utf-8"))
    assert interrupted.substage_runs == runner._RUN_SUBSTAGES[:12]
    invalid = interrupted.complete_substage(runner._RUN_SUBSTAGES[13])
    artifact.write_text(invalid.model_dump_json(), encoding="utf-8")

    async def unexpected_provider(*_args: object, **_kwargs: object) -> Document:
        pytest.fail("A cumulative substage gap must be rejected before provider work")

    monkeypatch.setattr(runner, "grow_roots", unexpected_provider)
    monkeypatch.setattr(runner, "validate_roots_citation_identity", unexpected_provider)
    with pytest.raises(ValueError, match="Saved evaluation checkpoint skips a substage"):
        asyncio.run(runner._run(data_root, results_root, None, run_dir))
    assert Document.model_validate_json(artifact.read_text(encoding="utf-8")) == invalid
    assert not (run_dir / "checkpoints").exists()


def test_resume_from_reporter_docket_checkpoint_skips_provider_requery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    workflow = importlib.import_module("mellea_lrc.workflows.validate_roots_citation_identity")
    calls: list[str] = []
    fail_review = True

    async def fake_grow(
        document: Document,
        *,
        hunt_dockets: bool,
        review_docket_roots: bool,
        field_extraction,
        checkpoint=None,
    ) -> Document:
        calls.append("grow")
        return _complete(document, runner._RUN_SUBSTAGES[:12])

    def retrieve(document: Document) -> Document:
        calls.append("12.1")
        return document.complete_substage(runner._RUN_SUBSTAGES[12])

    def review(document: Document) -> Document:
        nonlocal fail_review
        calls.append("13.1")
        if fail_review:
            fail_review = False
            raise RuntimeError("interrupted during reporter review")
        return document.complete_substage(runner._RUN_SUBSTAGES[14])

    def sync_stage(substage: str):
        def run(document: Document) -> Document:
            calls.append(substage)
            return document.complete_substage(substage)

        return run

    def async_stage(substage: str):
        async def run(document: Document) -> Document:
            calls.append(substage)
            return document.complete_substage(substage)

        return run

    async def body(
        document: Document,
        *,
        retrospective_date: date | None = None,
        courtlistener_client: object | None = None,
        checkpoint: Callable[[Document], None] | None = None,
    ) -> Document:
        assert document.substage_runs == _BODY_INPUT_SUBSTAGES
        return _complete_with_checkpoints(document, _BODY_SUBSTAGES, checkpoint)

    monkeypatch.setattr(runner, "grow_roots", fake_grow)
    _patch_operation(monkeypatch, "retrieve_reporter_root_clusters", retrieve)
    _patch_operation(monkeypatch, "judge_unique_reporter_roots_rule", review)
    for substage, name in (
        (
            "validate_roots_citation_identity.lookup_reporter_roots.retrieve_dockets",
            "retrieve_reporter_root_dockets",
        ),
        (
            "validate_roots_citation_identity.lookup_reporter_roots.judge_ambiguous_rule",
            "judge_ambiguous_reporter_roots_rule",
        ),
        (
            "validate_roots_citation_identity.lookup_docket_roots.retrieve_courtlistener_dockets",
            "retrieve_courtlistener_docket_roots",
        ),
        (
            "validate_roots_citation_identity.lookup_docket_roots.retrieve_govinfo_dockets",
            "retrieve_govinfo_docket_roots",
        ),
    ):
        _patch_operation(monkeypatch, name, sync_stage(substage))
    for substage, name in (
        (
            "validate_roots_citation_identity.lookup_reporter_roots.review_unique_llm",
            "review_unique_reporter_roots_llm",
        ),
        (
            "validate_roots_citation_identity.lookup_reporter_roots.review_ambiguous_llm",
            "review_ambiguous_reporter_roots_llm",
        ),
        (
            "validate_roots_citation_identity.lookup_docket_roots.review_courtlistener_dockets_llm",
            "review_courtlistener_docket_roots_llm",
        ),
        (
            "validate_roots_citation_identity.lookup_docket_roots.review_govinfo_dockets_llm",
            "review_govinfo_docket_roots_llm",
        ),
    ):
        _patch_operation(monkeypatch, name, async_stage(substage))
    monkeypatch.setattr(workflow, "corroborate_locator_bodies", body)

    with pytest.raises(RuntimeError, match="interrupted during reporter review"):
        asyncio.run(runner._run(data_root, tmp_path / "results", None))
    run_dir = next((tmp_path / "results").iterdir())
    checkpoint = run_dir / "documents" / "001.txt.json"
    assert checkpoint.exists()
    saved = Document.model_validate_json(checkpoint.read_text(encoding="utf-8"))
    assert saved.substage_runs == runner._RUN_SUBSTAGES[:14]

    assert asyncio.run(runner._run(tmp_path / "unused", tmp_path / "unused", None, run_dir)) == run_dir
    final = Document.model_validate_json((run_dir / "documents" / "001.txt.json").read_text())
    assert final.substage_runs == runner._RUN_SUBSTAGES
    assert final.get_substage(runner._RUN_SUBSTAGES[13]) == saved
    assert calls.count("grow") == 1
    assert calls.count("12.1") == 1
    assert calls.count(runner._RUN_SUBSTAGES[13]) == 1
    assert calls.count("13.1") == 2


def test_rewind_completed_document_to_reporter_retrieval_runs_later_reviews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    source = data_root / "primary" / "documents_txt" / "001.txt"
    checkpoint_dir = tmp_path / "completed-documents"
    checkpoint_dir.mkdir()
    completed = _complete(Document.from_source(source), runner._RUN_SUBSTAGES)
    (checkpoint_dir / "001.txt.json").write_text(completed.model_dump_json(), encoding="utf-8")
    workflow = importlib.import_module("mellea_lrc.workflows.validate_roots_citation_identity")
    calls: list[str] = []

    def unexpected_lookup(_document: Document) -> Document:
        pytest.fail("Saved 12.1 retrieval must not call the lookup provider")

    def sync_stage(substage: str):
        def run(document: Document) -> Document:
            calls.append(substage)
            return document.complete_substage(substage)

        return run

    def async_stage(substage: str):
        async def run(document: Document) -> Document:
            calls.append(substage)
            return document.complete_substage(substage)

        return run

    async def body(
        document: Document,
        *,
        retrospective_date: date | None = None,
        courtlistener_client: object | None = None,
        checkpoint: Callable[[Document], None] | None = None,
    ) -> Document:
        assert document.substage_runs == _BODY_INPUT_SUBSTAGES
        return _complete_with_checkpoints(document, _BODY_SUBSTAGES, checkpoint)

    _patch_operation(monkeypatch, "retrieve_reporter_root_clusters", unexpected_lookup)
    for substage, name in (
        (
            "validate_roots_citation_identity.lookup_reporter_roots.judge_unique_rule",
            "judge_unique_reporter_roots_rule",
        ),
        (
            "validate_roots_citation_identity.lookup_reporter_roots.retrieve_dockets",
            "retrieve_reporter_root_dockets",
        ),
        (
            "validate_roots_citation_identity.lookup_reporter_roots.judge_ambiguous_rule",
            "judge_ambiguous_reporter_roots_rule",
        ),
        (
            "validate_roots_citation_identity.lookup_docket_roots.retrieve_courtlistener_dockets",
            "retrieve_courtlistener_docket_roots",
        ),
        (
            "validate_roots_citation_identity.lookup_docket_roots.retrieve_govinfo_dockets",
            "retrieve_govinfo_docket_roots",
        ),
        (runner._FIELD_IDENTITY_SUBSTAGE, "judge_root_identity_from_fields"),
    ):
        _patch_operation(monkeypatch, name, sync_stage(substage))
    for substage, name in (
        (
            "validate_roots_citation_identity.lookup_reporter_roots.review_unique_llm",
            "review_unique_reporter_roots_llm",
        ),
        (
            "validate_roots_citation_identity.lookup_reporter_roots.review_ambiguous_llm",
            "review_ambiguous_reporter_roots_llm",
        ),
        (
            "validate_roots_citation_identity.lookup_docket_roots.review_courtlistener_dockets_llm",
            "review_courtlistener_docket_roots_llm",
        ),
        (
            "validate_roots_citation_identity.lookup_docket_roots.review_govinfo_dockets_llm",
            "review_govinfo_docket_roots_llm",
        ),
    ):
        _patch_operation(monkeypatch, name, async_stage(substage))
    monkeypatch.setattr(workflow, "corroborate_locator_bodies", body)

    run_dir = asyncio.run(
        runner._run(
            data_root,
            tmp_path / "results",
            None,
            from_checkpoint_documents=checkpoint_dir,
            checkpoint_substage=runner._RUN_SUBSTAGES[12],
        )
    )
    final = Document.model_validate_json((run_dir / "documents" / "001.txt.json").read_text())
    assert final.get_substage(runner._RUN_SUBSTAGES[12]) == completed.get_substage(runner._RUN_SUBSTAGES[12])
    assert final.substage_runs == runner._RUN_SUBSTAGES
    assert calls == list(runner._RUN_SUBSTAGES[13:_BODY_START])
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["checkpoint_substage"] == runner._RUN_SUBSTAGES[12]
    assert record["from_checkpoint_documents"] == str(checkpoint_dir)


@pytest.mark.parametrize("invalid", ["unsupported-substage", "non-prefix-document"])
def test_rewind_rejects_invalid_validation_checkpoint_before_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    source = data_root / "primary" / "documents_txt" / "001.txt"
    checkpoint_dir = tmp_path / "checkpoint-documents"
    checkpoint_dir.mkdir()
    selected_stage = runner._RUN_SUBSTAGES[12]
    if invalid == "unsupported-substage":
        document = _complete(Document.from_source(source), runner._RUN_SUBSTAGES)
        selected_stage = runner._RUN_SUBSTAGES[11]
        error = "Unsupported validation checkpoint"
    else:
        document = _complete(Document.from_source(source), (*runner._ROOT_SUBSTAGES, selected_stage))
        error = "Saved input checkpoint is incomplete"
    (checkpoint_dir / "001.txt.json").write_text(document.model_dump_json(), encoding="utf-8")

    async def unexpected_provider(*_args: object, **_kwargs: object) -> Document:
        pytest.fail("Invalid checkpoint must be rejected before provider work")

    monkeypatch.setattr(runner, "grow_roots", unexpected_provider)
    monkeypatch.setattr(runner, "validate_roots_citation_identity", unexpected_provider)
    with pytest.raises(ValueError, match=error):
        asyncio.run(
            runner._run(
                data_root,
                tmp_path / "results",
                None,
                from_checkpoint_documents=checkpoint_dir,
                checkpoint_substage=selected_stage,
            )
        )


def test_resume_rejects_stale_saved_source_before_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    roots_dir = tmp_path / "roots"
    roots_dir.mkdir()
    source = data_root / "primary" / "documents_txt" / "001.txt"
    roots = _complete(Document.from_source(source), runner._ROOT_SUBSTAGES)
    (roots_dir / "001.txt.json").write_text(roots.model_dump_json(), encoding="utf-8")
    source.write_text("changed source\n", encoding="utf-8")

    async def unexpected_review(_document: Document) -> Document:
        pytest.fail("A stale checkpoint must not reach the provider-backed review")

    root_stage = importlib.import_module("mellea_lrc.workflows.grow_roots.form_root_groups")
    monkeypatch.setattr(root_stage, "reassign_docket_roots_llm", unexpected_review)
    with pytest.raises(ValueError, match="Saved Document text differs"):
        asyncio.run(runner._run(data_root, tmp_path / "results", roots_dir))


def test_resume_from_reporter_llm_checkpoint_only_runs_later_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    checkpoint_dir = tmp_path / "reporter-review-input"
    checkpoint_dir.mkdir()
    source = data_root / "primary" / "documents_txt" / "001.txt"
    ready = _complete(Document.from_source(source), runner._REPORTER_REVIEW_INPUT_SUBSTAGES)
    (checkpoint_dir / "001.txt.json").write_text(ready.model_dump_json(), encoding="utf-8")
    calls: list[str] = []

    async def unique(document: Document) -> Document:
        calls.append("14")
        return document.complete_substage(runner._RUN_SUBSTAGES[16])

    async def ambiguous(document: Document) -> Document:
        calls.append("15")
        return document.complete_substage(runner._RUN_SUBSTAGES[17])

    def docket(document: Document) -> Document:
        calls.append("16")
        return document.complete_substage(runner._RUN_SUBSTAGES[18])

    async def docket_review(document: Document) -> Document:
        calls.append("17")
        return document.complete_substage(runner._RUN_SUBSTAGES[19])

    def govinfo(document: Document) -> Document:
        calls.append("18")
        return document.complete_substage(runner._RUN_SUBSTAGES[20])

    async def govinfo_review(document: Document) -> Document:
        calls.append("19")
        return document.complete_substage(runner._RUN_SUBSTAGES[21])

    monkeypatch.setattr(runner, "review_unique_reporter_roots_llm", unique)
    monkeypatch.setattr(runner, "review_ambiguous_reporter_roots_llm", ambiguous)
    monkeypatch.setattr(runner, "retrieve_courtlistener_docket_roots", docket)
    monkeypatch.setattr(runner, "review_courtlistener_docket_roots_llm", docket_review)
    monkeypatch.setattr(runner, "retrieve_govinfo_docket_roots", govinfo)
    monkeypatch.setattr(runner, "review_govinfo_docket_roots_llm", govinfo_review)
    run_dir = asyncio.run(runner._run(data_root, tmp_path / "results", None, None, checkpoint_dir))

    assert calls == ["14", "15", "16", "17", "18", "19"]
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["status"] == "complete"
    saved = Document.model_validate_json((run_dir / "documents" / "001.txt.json").read_text(encoding="utf-8"))
    assert saved.get_substage(runner._REPORTER_REVIEW_INPUT_SUBSTAGE) == ready
    assert asyncio.run(runner._run(tmp_path / "ignored", tmp_path / "unused", None, run_dir)) == run_dir
    assert calls == ["14", "15", "16", "17", "18", "19"]


def test_reporter_review_replay_reuses_saved_docket_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    source_path = data_root / "primary" / "documents_txt" / "001.txt"
    source_path.write_text("Acme v. Reed, Case No. 2:31-cv-45821 (S.D.N.Y. 2031).", encoding="utf-8")
    monkeypatch.setattr(
        importlib.import_module("mellea_lrc.extraction.hunt_docket_locators"),
        "suspected_dockets",
        lambda _: (),
    )
    ready = asyncio.run(grow_roots(Document.from_source(source_path), hunt_dockets=True)).get_substage(
        runner._ROOT_SUBSTAGE
    )
    ready = _complete(ready, runner._REPORTER_REVIEW_INPUT_SUBSTAGES[11:])
    root = next(root for root in ready.roots if isinstance(root, FullDocketCitation))
    prior = _complete(ready, runner._RUN_SUBSTAGES[16:18])
    recorded = root.record(runner._DOCKET_LOOKUP_SUBSTAGE)
    lookup = DocketLookup(
        node_id=recorded.nodes[-1].id,
        attempts=(DocketLookupAttempt(source_type="d", query="docketNumber:(2:31-cv-45821)"),),
    )
    prior = prior.replace_citation(recorded.with_docket_lookup(lookup)).complete_substage(
        runner._DOCKET_LOOKUP_SUBSTAGE
    )
    checkpoint_dir = tmp_path / "reporter-review-input"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "001.txt.json").write_text(prior.model_dump_json(), encoding="utf-8")

    async def unique(document: Document) -> Document:
        return document.complete_substage(runner._RUN_SUBSTAGES[16])

    async def ambiguous(document: Document) -> Document:
        return document.complete_substage(runner._RUN_SUBSTAGES[17])

    def unexpected_lookup(_document: Document) -> Document:
        pytest.fail("Saved docket lookup must be reused")

    async def docket_review(document: Document) -> Document:
        return document.complete_substage(runner._RUN_SUBSTAGES[19])

    def govinfo(document: Document) -> Document:
        return document.complete_substage(runner._RUN_SUBSTAGES[20])

    async def govinfo_review(document: Document) -> Document:
        return document.complete_substage(runner._RUN_SUBSTAGES[21])

    monkeypatch.setattr(runner, "review_unique_reporter_roots_llm", unique)
    monkeypatch.setattr(runner, "review_ambiguous_reporter_roots_llm", ambiguous)
    monkeypatch.setattr(runner, "retrieve_courtlistener_docket_roots", unexpected_lookup)
    monkeypatch.setattr(runner, "review_courtlistener_docket_roots_llm", docket_review)
    monkeypatch.setattr(runner, "retrieve_govinfo_docket_roots", govinfo)
    monkeypatch.setattr(runner, "review_govinfo_docket_roots_llm", govinfo_review)
    run_dir = asyncio.run(runner._run(data_root, tmp_path / "results", None, None, checkpoint_dir, True))

    saved = Document.model_validate_json((run_dir / "documents" / "001.txt.json").read_text(encoding="utf-8"))
    replayed_root = next(root for root in saved.roots if isinstance(root, FullDocketCitation))
    assert replayed_root.docket_lookup == lookup
    assert saved.get_substage(runner._REPORTER_REVIEW_INPUT_SUBSTAGE) == ready


def test_resume_from_docket_review_checkpoint_only_runs_govinfo_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    checkpoint_dir = tmp_path / "docket-review-input"
    checkpoint_dir.mkdir()
    source = data_root / "primary" / "documents_txt" / "001.txt"
    ready = _complete(Document.from_source(source), runner._DOCKET_REVIEW_INPUT_SUBSTAGES)
    (checkpoint_dir / "001.txt.json").write_text(ready.model_dump_json(), encoding="utf-8")
    calls: list[str] = []

    def govinfo(document: Document) -> Document:
        calls.append("18")
        return document.complete_substage(runner._RUN_SUBSTAGES[20])

    async def govinfo_review(document: Document) -> Document:
        calls.append("19")
        return document.complete_substage(runner._RUN_SUBSTAGES[21])

    monkeypatch.setattr(runner, "retrieve_govinfo_docket_roots", govinfo)
    monkeypatch.setattr(runner, "review_govinfo_docket_roots_llm", govinfo_review)
    run_dir = asyncio.run(
        runner._run(data_root, tmp_path / "results", None, None, None, False, checkpoint_dir)
    )

    assert calls == ["18", "19"]
    saved = Document.model_validate_json((run_dir / "documents" / "001.txt.json").read_text(encoding="utf-8"))
    assert saved.substage_runs == runner._RUN_SUBSTAGES
    assert saved.get_substage(runner._DOCKET_REVIEW_INPUT_SUBSTAGE) == ready


def test_resume_from_validation_checkpoint_runs_aggregation_then_body_stages_and_reuses_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filenames = ("001.txt", "002.txt")
    data_root = _dataset(tmp_path, filenames)
    checkpoint_dir = tmp_path / "validation-input"
    checkpoint_dir.mkdir()
    for filename in filenames:
        source = data_root / "primary" / "documents_txt" / filename
        ready = _complete(Document.from_source(source), runner._VALIDATION_INPUT_SUBSTAGES)
        (checkpoint_dir / f"{filename}.json").write_text(ready.model_dump_json(), encoding="utf-8")
    calls: list[tuple[str, date | None]] = []

    async def body(
        document: Document,
        *,
        retrospective_date: date | None = None,
        courtlistener_client: object | None = None,
        checkpoint: Callable[[Document], None] | None = None,
    ) -> Document:
        calls.append((Path(document.source_path or "").name, retrospective_date))
        assert document.substage_runs in (_BODY_INPUT_SUBSTAGES, runner._RUN_SUBSTAGES)
        return _complete_with_checkpoints(document, _BODY_SUBSTAGES, checkpoint)

    monkeypatch.setattr(_ROOT_WORKFLOW, "corroborate_locator_bodies", body)
    cutoff = date(2024, 6, 1)
    run_dir = asyncio.run(
        runner._run(
            data_root,
            tmp_path / "results",
            None,
            from_validation_documents=checkpoint_dir,
            retrospective_date=cutoff,
        )
    )
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["from_validation_documents"] == str(checkpoint_dir)
    assert record["retrospective_date"] == cutoff.isoformat()
    assert calls == [(filename, cutoff) for filename in filenames]
    for filename in filenames:
        saved = Document.model_validate_json((run_dir / "documents" / f"{filename}.json").read_text())
        assert saved.substage_runs == runner._RUN_SUBSTAGES
        assert saved.get_substage(runner._VALIDATION_INPUT_SUBSTAGE) == Document.model_validate_json(
            (checkpoint_dir / f"{filename}.json").read_text()
        )

    (run_dir / "documents" / "002.txt.json").unlink()
    record["status"] = "failed"
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    assert asyncio.run(runner._run(tmp_path / "unused", tmp_path / "unused", None, run_dir)) == run_dir
    assert calls == [(filename, cutoff) for filename in filenames] + [("002.txt", cutoff)]


def test_resume_after_locator_body_review_failure_reuses_all_retrieval_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    source = data_root / "primary" / "documents_txt" / "001.txt"
    checkpoint_dir = tmp_path / "validation-input"
    checkpoint_dir.mkdir()
    ready = _complete(Document.from_source(source), runner._VALIDATION_INPUT_SUBSTAGES)
    (checkpoint_dir / "001.txt.json").write_text(ready.model_dump_json(), encoding="utf-8")
    workflow = importlib.import_module("mellea_lrc.workflows.validate_roots_citation_identity")
    monkeypatch.setattr(workflow, "corroborate_locator_bodies", _LOCATOR_REVIEW)
    calls: list[str] = []
    fail_review = True

    def retrieve(substage: str):
        def run(document: Document, *, retrospective_date: date | None = None) -> Document:
            calls.append(substage)
            return document.complete_substage(substage)

        return run

    async def review(document: Document) -> Document:
        nonlocal fail_review
        calls.append(runner._LOCATOR_BODY_REVIEW_SUBSTAGE)
        if fail_review:
            fail_review = False
            raise RuntimeError("interrupted during locator-body review")
        return document.complete_substage(runner._LOCATOR_BODY_REVIEW_SUBSTAGE)

    _patch_operation(
        monkeypatch,
        "retrieve_courtlistener_locator_opinions",
        retrieve(runner._COURTLISTENER_OPINION_SUBSTAGE),
    )
    _patch_operation(
        monkeypatch,
        "retrieve_courtlistener_locator_filings",
        retrieve(runner._COURTLISTENER_RECAP_SUBSTAGE),
    )
    _patch_operation(
        monkeypatch,
        "retrieve_govinfo_locator_opinions",
        retrieve(runner._GOVINFO_OPINION_SUBSTAGE),
    )
    _patch_operation(monkeypatch, "review_locator_bodies_llm", review)

    with pytest.raises(RuntimeError, match="interrupted during locator-body review"):
        asyncio.run(
            runner._run(data_root, tmp_path / "results", None, from_validation_documents=checkpoint_dir)
        )
    run_dir = next((tmp_path / "results").iterdir())
    stage22 = Document.model_validate_json((run_dir / "documents" / "001.txt.json").read_text())
    assert stage22.substage_runs[-1] == runner._GOVINFO_OPINION_SUBSTAGE
    assert runner._LOCATOR_BODY_REVIEW_SUBSTAGE not in stage22.substage_runs
    for substage in runner._BODY_SEARCH_SUBSTAGES:
        assert (
            stage22.get_substage(substage).substage_runs
            == runner._RUN_SUBSTAGES[: runner._RUN_SUBSTAGES.index(substage) + 1]
        )
    assert not (run_dir / "checkpoints").exists()
    assert calls == [*runner._BODY_SEARCH_SUBSTAGES, runner._LOCATOR_BODY_REVIEW_SUBSTAGE]

    assert asyncio.run(runner._run(tmp_path / "unused", tmp_path / "unused", None, run_dir)) == run_dir
    final = Document.model_validate_json((run_dir / "documents" / "001.txt.json").read_text())
    assert final.substage_runs == runner._RUN_SUBSTAGES
    assert final.get_substage(runner._GOVINFO_OPINION_SUBSTAGE) == stage22
    assert calls == [
        *runner._BODY_SEARCH_SUBSTAGES,
        runner._LOCATOR_BODY_REVIEW_SUBSTAGE,
        runner._LOCATOR_BODY_REVIEW_SUBSTAGE,
    ]


def test_rewind_completed_document_to_stage22_replays_only_locator_body_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    source = data_root / "primary" / "documents_txt" / "001.txt"
    checkpoint_dir = tmp_path / "completed-documents"
    checkpoint_dir.mkdir()
    completed = _complete(Document.from_source(source), runner._RUN_SUBSTAGES)
    (checkpoint_dir / "001.txt.json").write_text(completed.model_dump_json(), encoding="utf-8")
    workflow = importlib.import_module("mellea_lrc.workflows.validate_roots_citation_identity")
    monkeypatch.setattr(workflow, "corroborate_locator_bodies", _LOCATOR_REVIEW)
    calls: list[str] = []

    def unexpected_retrieval(*_args: object, **_kwargs: object) -> Document:
        pytest.fail("Stage 22 replay must reuse saved locator-body retrievals")

    async def review(document: Document) -> Document:
        calls.append(runner._LOCATOR_BODY_REVIEW_SUBSTAGE)
        assert document == completed.get_substage(runner._GOVINFO_OPINION_SUBSTAGE)
        return document.complete_substage(runner._LOCATOR_BODY_REVIEW_SUBSTAGE)

    _patch_operation(monkeypatch, "retrieve_courtlistener_locator_opinions", unexpected_retrieval)
    _patch_operation(monkeypatch, "retrieve_courtlistener_locator_filings", unexpected_retrieval)
    _patch_operation(monkeypatch, "retrieve_govinfo_locator_opinions", unexpected_retrieval)
    _patch_operation(monkeypatch, "review_locator_bodies_llm", review)

    results_root = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python -m evaluations",
            "--data-root",
            str(data_root),
            "--results-root",
            str(results_root),
            "--from-checkpoint-documents",
            str(checkpoint_dir),
            "--checkpoint-substage",
            runner._GOVINFO_OPINION_SUBSTAGE,
        ],
    )
    runner.main()
    run_dir = next(results_root.iterdir())
    final = Document.model_validate_json((run_dir / "documents" / "001.txt.json").read_text())
    assert final == completed
    assert calls == [runner._LOCATOR_BODY_REVIEW_SUBSTAGE]
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["checkpoint_substage"] == runner._GOVINFO_OPINION_SUBSTAGE
    assert record["from_checkpoint_documents"] == str(checkpoint_dir)


def test_locator_review_replay_saves_each_stage_and_resumes_without_earlier_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filenames = ("001.txt", "002.txt")
    data_root = _dataset(tmp_path, filenames)
    _annotation_cutoffs(data_root, {"001.txt": "2024-01-02", "002.txt": "2025-03-04"})
    checkpoint_dir = tmp_path / "locator-review-input"
    checkpoint_dir.mkdir()
    for filename in filenames:
        source = data_root / "primary" / "documents_txt" / filename
        stage23 = _complete(Document.from_source(source), runner._RUN_SUBSTAGES)
        (checkpoint_dir / f"{filename}.json").write_text(stage23.model_dump_json(), encoding="utf-8")

    calls: list[tuple[str, str, date | None]] = []
    fail_once = True

    def provider(substage: str):
        def run(document: Document, *, retrospective_date: date | None = None) -> Document:
            nonlocal fail_once
            name = Path(document.source_path or "").name
            calls.append((name, substage, retrospective_date))
            if name == "002.txt" and substage == runner._FIELD_SUBSTAGES[2] and fail_once:
                fail_once = False
                raise RuntimeError("interrupted after substage 25")
            return document.complete_substage(substage)

        return run

    async def review(document: Document) -> Document:
        calls.append((Path(document.source_path or "").name, runner._FIELD_SUBSTAGES[3], None))
        return document.complete_substage(runner._FIELD_SUBSTAGES[3])

    monkeypatch.setattr(
        runner, "retrieve_courtlistener_intended_case_opinions", provider(runner._FIELD_SUBSTAGES[0])
    )
    monkeypatch.setattr(
        runner, "retrieve_courtlistener_intended_case_filings", provider(runner._FIELD_SUBSTAGES[1])
    )
    monkeypatch.setattr(
        runner, "retrieve_govinfo_intended_case_opinions", provider(runner._FIELD_SUBSTAGES[2])
    )
    monkeypatch.setattr(runner, "review_intended_cases_llm", review)

    with pytest.raises(RuntimeError, match="interrupted after substage 25"):
        asyncio.run(
            runner._run(
                data_root,
                tmp_path / "results",
                None,
                annotation_case_cutoffs=True,
                from_locator_review_documents=checkpoint_dir,
            )
        )
    run_dir = next((tmp_path / "results").iterdir())
    assert json.loads((run_dir / "run.json").read_text())["status"] == "failed"
    interrupted = Document.model_validate_json((run_dir / "documents" / "002.txt.json").read_text())
    assert interrupted.substage_runs[-1] == runner._FIELD_SUBSTAGES[1]
    assert runner._FIELD_SUBSTAGES[2] not in interrupted.substage_runs
    assert not (run_dir / "checkpoints").exists()

    assert asyncio.run(runner._run(tmp_path / "ignored", tmp_path / "unused", None, run_dir)) == run_dir
    assert json.loads((run_dir / "run.json").read_text())["status"] == "complete"
    assert [(name, substage) for name, substage, _ in calls].count(
        ("002.txt", runner._FIELD_SUBSTAGES[0])
    ) == 1
    assert [(name, substage) for name, substage, _ in calls].count(
        ("002.txt", runner._FIELD_SUBSTAGES[1])
    ) == 1
    assert [(name, substage) for name, substage, _ in calls].count(
        ("002.txt", runner._FIELD_SUBSTAGES[2])
    ) == 2
    assert [
        cutoff
        for name, substage, cutoff in calls
        if name == "002.txt" and substage != runner._FIELD_SUBSTAGES[3]
    ] == [
        date(2025, 3, 4),
        date(2025, 3, 4),
        date(2025, 3, 4),
        date(2025, 3, 4),
    ]
    for filename in filenames:
        final = Document.model_validate_json((run_dir / "documents" / f"{filename}.json").read_text())
        original = Document.model_validate_json((checkpoint_dir / f"{filename}.json").read_text())
        assert final.substage_runs == runner._FIELD_RUN_SUBSTAGES
        assert final.get_stage("validate_roots_citation_identity.corroborate_locator_bodies") == original
        for substage in runner._FIELD_SUBSTAGES:
            assert (
                final.get_substage(substage).substage_runs
                == runner._FIELD_RUN_SUBSTAGES[: runner._FIELD_RUN_SUBSTAGES.index(substage) + 1]
            )
        if filename == "002.txt":
            assert final.get_substage(runner._FIELD_SUBSTAGES[1]) == interrupted


def test_transient_field_search_stops_before_later_providers_and_retries_its_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    source = data_root / "primary" / "documents_txt" / "001.txt"
    source.write_text("Brown v. Board of Education, 347 U.S. 483 (1954).", encoding="utf-8")
    stage23 = asyncio.run(
        grow_roots(Document.from_source(source), hunt_dockets=True, review_docket_roots=True)
    )
    stage23 = _complete(stage23, runner._RUN_SUBSTAGES[12:-1])
    root = stage23.roots[0].record(runner._LOCATOR_BODY_REVIEW_SUBSTAGE)
    stage23 = stage23.replace_citation(
        root.with_route(
            "validate_roots_citation_identity.discover_intended_cases.retrieve_courtlistener_opinions"
        )
    ).complete_substage(runner._LOCATOR_BODY_REVIEW_SUBSTAGE)
    stage23 = complete_stage_boundary(stage23, runner._FIELD_RUN_SUBSTAGES)
    checkpoint_dir = tmp_path / "locator-review-input"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "001.txt.json").write_text(stage23.model_dump_json(), encoding="utf-8")

    calls: list[str] = []

    def opinion(document: Document, *, retrospective_date: date | None = None) -> Document:
        calls.append("24")
        recorded = document.roots[0].record(runner._FIELD_SUBSTAGES[0])
        result = FieldBodySearch(
            node_id=recorded.nodes[-1].id,
            source=BodySource.COURTLISTENER_OPINION,
            retrospective_date=retrospective_date,
            query_name="Brown v. Board of Education",
            failures=(
                (BodyEvidenceFailure(failure_type="transport_error", message="Temporary failure"),)
                if len(calls) == 1
                else ()
            ),
        )
        return document.replace_citation(recorded.with_field_body_search(result)).complete_substage(
            runner._FIELD_SUBSTAGES[0]
        )

    def later(substage: str):
        def run(document: Document, *, retrospective_date: date | None = None) -> Document:
            calls.append(substage.rsplit(".", 1)[-1])
            return document.complete_substage(substage)

        return run

    async def review(document: Document) -> Document:
        calls.append("27")
        return document.complete_substage(runner._FIELD_SUBSTAGES[3])

    monkeypatch.setattr(runner, "retrieve_courtlistener_intended_case_opinions", opinion)
    monkeypatch.setattr(
        runner, "retrieve_courtlistener_intended_case_filings", later(runner._FIELD_SUBSTAGES[1])
    )
    monkeypatch.setattr(runner, "retrieve_govinfo_intended_case_opinions", later(runner._FIELD_SUBSTAGES[2]))
    monkeypatch.setattr(runner, "review_intended_cases_llm", review)

    with pytest.raises(RuntimeError, match="transient provider failure"):
        asyncio.run(
            runner._run(
                data_root,
                tmp_path / "results",
                None,
                from_locator_review_documents=checkpoint_dir,
            )
        )
    run_dir = next((tmp_path / "results").iterdir())
    assert calls == ["24"]
    interrupted = Document.model_validate_json((run_dir / "documents" / "001.txt.json").read_text())
    assert interrupted.substage_runs[-1] == runner._FIELD_SUBSTAGES[0]
    assert runner._FIELD_SUBSTAGES[1] not in interrupted.substage_runs
    assert not (run_dir / "checkpoints").exists()

    asyncio.run(runner._run(tmp_path / "ignored", tmp_path / "unused", None, run_dir))
    assert calls == ["24", "24", "retrieve_courtlistener_filings", "retrieve_govinfo_opinions", "27"]
    final = Document.model_validate_json((run_dir / "documents" / "001.txt.json").read_text())
    assert final.substage_runs == runner._FIELD_RUN_SUBSTAGES
    assert not final.roots[0].field_body_searches[0].failures


def test_annotation_headers_route_case_cutoffs_and_are_verified_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filenames = ("001.txt", "002.txt", "003.txt")
    data_root = _dataset(tmp_path, filenames)
    checkpoint_dir = tmp_path / "validation-input"
    checkpoint_dir.mkdir()
    for filename in filenames:
        source = data_root / "primary" / "documents_txt" / filename
        ready = _complete(Document.from_source(source), runner._VALIDATION_INPUT_SUBSTAGES)
        (checkpoint_dir / f"{filename}.json").write_text(ready.model_dump_json(), encoding="utf-8")
    filing_dates = {"001.txt": "2024-01-02", "002.txt": "2025-03-04", "003.txt": "2026-05-06"}
    _annotation_cutoffs(
        data_root,
        filing_dates,
        {"001.txt": "001.txt", "002.txt": "001.txt", "003.txt": "003.txt"},
    )
    expected = {"001.txt": "2024-01-02", "002.txt": "2024-01-02", "003.txt": "2026-05-06"}
    calls: list[tuple[str, date | None]] = []

    async def body(
        document: Document,
        *,
        retrospective_date: date | None = None,
        courtlistener_client: object | None = None,
        checkpoint: Callable[[Document], None] | None = None,
    ) -> Document:
        calls.append((Path(document.source_path or "").name, retrospective_date))
        assert document.substage_runs == _BODY_INPUT_SUBSTAGES
        return _complete_with_checkpoints(document, _BODY_SUBSTAGES, checkpoint)

    monkeypatch.setattr(_ROOT_WORKFLOW, "corroborate_locator_bodies", body)
    run_dir = asyncio.run(
        runner._run(
            data_root,
            tmp_path / "results",
            None,
            from_validation_documents=checkpoint_dir,
            annotation_case_cutoffs=True,
        )
    )
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["annotation_case_cutoffs"] is True
    assert record["case_cutoffs"] == expected
    assert set(record["annotation_header_sha256"]) == set(filenames)
    assert all(len(digest) == 64 for digest in record["annotation_header_sha256"].values())
    assert calls == [(name, date.fromisoformat(expected[name])) for name in filenames]

    (run_dir / "documents" / "002.txt.json").unlink()
    record["status"] = "failed"
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    assert asyncio.run(runner._run(tmp_path / "ignored", tmp_path / "unused", None, run_dir)) == run_dir
    assert calls[-1] == ("002.txt", date(2024, 1, 2))

    header_path = data_root / "primary" / "documents" / "001.jsonl"
    header = json.loads(header_path.read_text(encoding="utf-8"))
    header["filing"]["provenance"]["evidence"] = "Changed evidence"
    header_path.write_text(json.dumps(header) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Annotation case cutoffs differ"):
        asyncio.run(runner._run(tmp_path / "ignored", tmp_path / "unused", None, run_dir))
    assert len(calls) == 4


@pytest.mark.parametrize(
    "headers,error",
    [
        ({"001.txt": {"filing": {"date": "2024-02-30"}}}, "invalid filing or cutoff date"),
        ({"001.txt": {"filing": {"provenance": ""}}}, "sourced PDF evidence"),
        (
            {"001.txt": {"case_cutoff": {"date": "2025-03-04", "source_document": "001.txt"}}},
            "earliest sampled filing",
        ),
        (
            {"001.txt": {"case_cutoff": {"date": "2024-01-02", "source_document": "missing.txt"}}},
            "not a sampled filing",
        ),
    ],
)
def test_annotation_cutoffs_reject_invalid_headers_before_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, headers: dict[str, object], error: str
) -> None:
    data_root = _dataset(tmp_path, ("001.txt", "002.txt"))
    _annotation_cutoffs(data_root, {"001.txt": "2024-01-02", "002.txt": "2025-03-04"})
    for filename, edits in headers.items():
        path = data_root / "primary" / "documents" / f"{Path(filename).stem}.jsonl"
        header = json.loads(path.read_text(encoding="utf-8"))
        for key, value in edits.items():
            header[key].update(value)
        path.write_text(json.dumps(header) + "\n", encoding="utf-8")

    async def unexpected_grow(*_args: object, **_kwargs: object) -> Document:
        pytest.fail("Provider-backed work must not start with invalid filing dates")

    monkeypatch.setattr(runner, "grow_roots", unexpected_grow)
    with pytest.raises(ValueError, match=error):
        asyncio.run(runner._run(data_root, tmp_path / "results", None, annotation_case_cutoffs=True))
    assert not (tmp_path / "results").exists()


def test_resume_rejects_saved_body_search_from_a_different_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    _annotation_cutoffs(data_root, {"001.txt": "2024-06-01"})
    source_path = data_root / "primary" / "documents_txt" / "001.txt"
    source = "Acme v. Reed, Case No. 2:31-cv-45821 (S.D.N.Y. 2031)."
    source_path.write_text(source, encoding="utf-8")
    ready = Document.from_source(source_path).complete_substage(runner._RUN_SUBSTAGES[0])
    locator = "Case No. 2:31-cv-45821"
    number = "2:31-cv-45821"
    root = FullDocketCitation.from_locator(
        citation_id="docket:0",
        substage=runner._RUN_SUBSTAGES[1],
        source=source,
        span=Span(source.index(locator), source.index(locator) + len(locator)),
        number_span=Span(source.index(number), source.index(number) + len(number)),
    )
    ready = ready.add_citation(root).complete_substage(runner._RUN_SUBSTAGES[1])
    ready = _complete(ready, runner._RUN_SUBSTAGES[2:10])
    ready = ready.replace_citation(root.record(runner._ROOT_SUBSTAGE).with_root(root.id)).complete_substage(
        runner._ROOT_SUBSTAGE
    )
    ready = _complete(ready, runner._RUN_SUBSTAGES[11:_BODY_START])
    checkpoint_dir = tmp_path / "validation-input"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "001.txt.json").write_text(ready.model_dump_json(), encoding="utf-8")

    async def body(
        document: Document,
        *,
        retrospective_date: date | None = None,
        courtlistener_client: object | None = None,
        checkpoint: Callable[[Document], None] | None = None,
    ) -> Document:
        assert retrospective_date == date(2024, 6, 1)
        citation = document.roots[0].record(runner._COURTLISTENER_OPINION_SUBSTAGE)
        search = BodySearch(
            node_id=citation.nodes[-1].id,
            source=BodySource.COURTLISTENER_OPINION,
            retrospective_date=retrospective_date,
            attempts=(BodySearchAttempt(query=number),),
        )
        document = document.replace_citation(citation.with_body_search(search)).complete_substage(
            runner._COURTLISTENER_OPINION_SUBSTAGE
        )
        if checkpoint is not None:
            checkpoint(document)
        return _complete_with_checkpoints(document, runner._RUN_SUBSTAGES[_BODY_START + 1 :], checkpoint)

    monkeypatch.setattr(_ROOT_WORKFLOW, "corroborate_locator_bodies", body)
    run_dir = asyncio.run(
        runner._run(
            data_root,
            tmp_path / "results",
            None,
            from_validation_documents=checkpoint_dir,
            annotation_case_cutoffs=True,
        )
    )
    artifact = run_dir / "documents" / "001.txt.json"
    altered = json.loads(artifact.read_text(encoding="utf-8"))
    altered["citations"][0]["body_searches"][0]["retrospective_date"] = None
    artifact.write_text(json.dumps(altered), encoding="utf-8")

    with pytest.raises(ValueError, match="Saved body search cutoff differs"):
        asyncio.run(runner._run(tmp_path / "ignored", tmp_path / "unused", None, run_dir))


def test_reserved_pool_is_saved_reused_and_closed_without_saving_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filenames = ("001.txt", "002.txt")
    data_root = _dataset(tmp_path, filenames)
    checkpoint_dir = tmp_path / "validation-input"
    checkpoint_dir.mkdir()
    for filename in filenames:
        source = data_root / "primary" / "documents_txt" / filename
        ready = _complete(Document.from_source(source), runner._VALIDATION_INPUT_SUBSTAGES)
        (checkpoint_dir / f"{filename}.json").write_text(ready.model_dump_json(), encoding="utf-8")

    _courtlistener_settings(tmp_path, monkeypatch)
    calls: list[tuple[str, CourtListenerClient]] = []
    closed: list[CourtListenerClient] = []
    original_close = CourtListenerClient.close

    def close(client: CourtListenerClient) -> None:
        closed.append(client)
        original_close(client)

    async def body(
        document: Document,
        *,
        retrospective_date: date | None,
        courtlistener_client: CourtListenerClient,
        checkpoint: Callable[[Document], None] | None = None,
    ) -> Document:
        assert retrospective_date is None
        calls.append((Path(document.source_path or "").name, courtlistener_client))
        assert document.substage_runs == _BODY_INPUT_SUBSTAGES
        return _complete_with_checkpoints(document, _BODY_SUBSTAGES, checkpoint)

    monkeypatch.setattr(CourtListenerClient, "close", close)
    monkeypatch.setattr(_ROOT_WORKFLOW, "corroborate_locator_bodies", body)
    run_dir = asyncio.run(
        runner._run(
            data_root,
            tmp_path / "results",
            None,
            from_validation_documents=checkpoint_dir,
            courtlistener_pool="reserved",
        )
    )

    record_text = (run_dir / "run.json").read_text(encoding="utf-8")
    assert json.loads(record_text)["courtlistener_pool"] == "reserved"
    assert "offline-reserved-token" not in record_text
    assert [filename for filename, _client in calls] == list(filenames)
    assert calls[0][1] is calls[1][1]
    assert closed == [calls[0][1]]
    assert calls[0][1].config.base_url == "https://proxy.example/api/rest/v4/"
    assert calls[0][1].config.timeout_seconds == 47
    assert calls[0][1].config.pool == "reserved"
    assert calls[0][1].config.token == "offline-reserved-token"

    (run_dir / "documents" / "002.txt.json").unlink()
    assert asyncio.run(runner._run(tmp_path / "unused", tmp_path / "unused", None, run_dir)) == run_dir
    assert [filename for filename, _client in calls] == [*filenames, "002.txt"]
    assert calls[2][1] is not calls[0][1]
    assert calls[2][1].config.timeout_seconds == 47
    assert closed == [calls[0][1], calls[2][1]]


@pytest.mark.parametrize("reserved_setting", ["", "COURTLISTENER_API_TOKEN_RESERVED=\n"])
def test_reserved_pool_requires_file_token_before_creating_client_or_running_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reserved_setting: str
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    checkpoint_dir = tmp_path / "validation-input"
    checkpoint_dir.mkdir()
    source = data_root / "primary" / "documents_txt" / "001.txt"
    ready = _complete(Document.from_source(source), runner._VALIDATION_INPUT_SUBSTAGES)
    (checkpoint_dir / "001.txt.json").write_text(ready.model_dump_json(), encoding="utf-8")
    env_path = _courtlistener_settings(tmp_path, monkeypatch)
    env_path.write_text(
        env_path.read_text(encoding="utf-8").replace(
            "COURTLISTENER_API_TOKEN_RESERVED=offline-reserved-token\n", reserved_setting
        ),
        encoding="utf-8",
    )

    def unexpected_client(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Missing reserved credentials must fail before creating a provider client")

    async def unexpected_stage(*_args: object, **_kwargs: object) -> Document:
        pytest.fail("Missing reserved credentials must fail before running a stage")

    monkeypatch.setattr(runner, "CourtListenerClient", unexpected_client)
    monkeypatch.setattr(_ROOT_WORKFLOW, "corroborate_locator_bodies", unexpected_stage)
    with pytest.raises(RuntimeError, match="COURTLISTENER_API_TOKEN_RESERVED"):
        asyncio.run(
            runner._run(
                data_root,
                tmp_path / "results",
                None,
                from_validation_documents=checkpoint_dir,
                courtlistener_pool="reserved",
            )
        )
    run_dir = next((tmp_path / "results").iterdir())
    assert not list((run_dir / "documents").iterdir())
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_resume_existing_run_can_select_and_save_reserved_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    checkpoint_dir = tmp_path / "validation-input"
    checkpoint_dir.mkdir()
    source = data_root / "primary" / "documents_txt" / "001.txt"
    ready = _complete(Document.from_source(source), runner._VALIDATION_INPUT_SUBSTAGES)
    (checkpoint_dir / "001.txt.json").write_text(ready.model_dump_json(), encoding="utf-8")
    _courtlistener_settings(tmp_path, monkeypatch)
    clients: list[CourtListenerClient | None] = []

    async def body(
        document: Document,
        *,
        retrospective_date: date | None,
        courtlistener_client: CourtListenerClient | None = None,
        checkpoint: Callable[[Document], None] | None = None,
    ) -> Document:
        clients.append(courtlistener_client)
        assert document.substage_runs == _BODY_INPUT_SUBSTAGES
        return _complete_with_checkpoints(document, _BODY_SUBSTAGES, checkpoint)

    monkeypatch.setattr(_ROOT_WORKFLOW, "corroborate_locator_bodies", body)
    run_dir = asyncio.run(
        runner._run(data_root, tmp_path / "results", None, from_validation_documents=checkpoint_dir)
    )
    record_path = run_dir / "run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record.pop("courtlistener_pool")  # Older runs did not record a pool.
    record_path.write_text(json.dumps(record), encoding="utf-8")
    (run_dir / "documents" / "001.txt.json").unlink()

    assert (
        asyncio.run(
            runner._run(
                tmp_path / "unused",
                tmp_path / "unused",
                None,
                run_dir,
                courtlistener_pool="reserved",
            )
        )
        == run_dir
    )
    assert clients[0] is None
    assert clients[1] is not None
    assert clients[1].config.pool == "reserved"
    record_text = record_path.read_text(encoding="utf-8")
    assert json.loads(record_text)["courtlistener_pool"] == "reserved"
    assert "offline-reserved-token" not in record_text

    (run_dir / "documents" / "001.txt.json").unlink()
    assert asyncio.run(runner._run(tmp_path / "unused", tmp_path / "unused", None, run_dir)) == run_dir
    assert clients[2] is not None
    assert clients[2].config.pool == "reserved"


def test_resume_can_switch_reserved_run_to_proxy_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    checkpoint_dir = tmp_path / "validation-input"
    checkpoint_dir.mkdir()
    source = data_root / "primary" / "documents_txt" / "001.txt"
    ready = _complete(Document.from_source(source), runner._VALIDATION_INPUT_SUBSTAGES)
    (checkpoint_dir / "001.txt.json").write_text(ready.model_dump_json(), encoding="utf-8")
    _courtlistener_settings(tmp_path, monkeypatch)
    clients: list[CourtListenerClient] = []

    async def body(
        document: Document,
        *,
        retrospective_date: date | None,
        courtlistener_client: CourtListenerClient,
        checkpoint: Callable[[Document], None] | None = None,
    ) -> Document:
        clients.append(courtlistener_client)
        assert document.substage_runs == _BODY_INPUT_SUBSTAGES
        return _complete_with_checkpoints(document, _BODY_SUBSTAGES, checkpoint)

    monkeypatch.setattr(_ROOT_WORKFLOW, "corroborate_locator_bodies", body)
    run_dir = asyncio.run(
        runner._run(
            data_root,
            tmp_path / "results",
            None,
            from_validation_documents=checkpoint_dir,
            courtlistener_pool="reserved",
        )
    )
    (run_dir / "documents" / "001.txt.json").unlink()
    assert (
        asyncio.run(
            runner._run(tmp_path / "unused", tmp_path / "unused", None, run_dir, courtlistener_pool="proxy")
        )
        == run_dir
    )
    assert clients[0].config.pool == "reserved"
    assert clients[1].config.pool is None
    assert clients[1].config.token is None
    assert clients[1].config.base_url == "https://proxy.example/api/rest/v4/"
    assert clients[1].config.timeout_seconds == 47
    record_text = (run_dir / "run.json").read_text(encoding="utf-8")
    assert json.loads(record_text)["courtlistener_pool_history"] == ["reserved", "proxy"]
    assert json.loads(record_text)["courtlistener_pool"] == "proxy"
    assert "offline-reserved-token" not in record_text


def test_retry_body_substages_passes_selected_client_to_both_courtlistener_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _complete(Document.from_source("source"), _BODY_INPUT_SUBSTAGES)
    selected_client = object()
    calls: list[str] = []

    def opinion(document: Document, *, retrospective_date: date | None, client: object) -> Document:
        assert client is selected_client
        assert document.substage_runs == _BODY_INPUT_SUBSTAGES
        calls.append("opinion")
        return document.complete_substage(runner._COURTLISTENER_OPINION_SUBSTAGE)

    def recap(document: Document, *, retrospective_date: date | None, client: object) -> Document:
        assert client is selected_client
        calls.append("recap")
        return document.complete_substage(runner._COURTLISTENER_RECAP_SUBSTAGE)

    def govinfo(document: Document, *, retrospective_date: date | None) -> Document:
        calls.append("govinfo")
        return document.complete_substage(runner._GOVINFO_OPINION_SUBSTAGE)

    async def review(document: Document) -> Document:
        calls.append("review")
        return document.complete_substage(runner._LOCATOR_BODY_REVIEW_SUBSTAGE)

    monkeypatch.setattr(runner, "retrieve_courtlistener_locator_opinions", opinion)
    monkeypatch.setattr(runner, "retrieve_courtlistener_locator_filings", recap)
    monkeypatch.setattr(runner, "retrieve_govinfo_locator_opinions", govinfo)
    monkeypatch.setattr(runner, "review_locator_bodies_llm", review)
    result = asyncio.run(
        runner._retry_body_substages(document, runner._COURTLISTENER_OPINION_SUBSTAGE, None, selected_client)
    )
    assert result.substage_runs == runner._RUN_SUBSTAGES
    assert calls == ["opinion", "recap", "govinfo", "review"]


@pytest.mark.parametrize("aggregation_complete", [False, True])
def test_validation_workflow_passes_selected_client_after_aggregation_to_both_body_stages(
    monkeypatch: pytest.MonkeyPatch,
    aggregation_complete: bool,
) -> None:
    workflow = importlib.import_module("mellea_lrc.workflows.validate_roots_citation_identity")
    monkeypatch.setattr(workflow, "corroborate_locator_bodies", _LOCATOR_REVIEW)
    input_substages = _BODY_INPUT_SUBSTAGES if aggregation_complete else runner._VALIDATION_INPUT_SUBSTAGES
    document = _complete(Document.from_source("source"), input_substages)
    selected_client = object()
    calls: list[str] = []
    checkpoints: list[Document] = []

    def opinion(document: Document, *, retrospective_date: date | None, client: object) -> Document:
        assert client is selected_client
        assert document.substage_runs == _BODY_INPUT_SUBSTAGES
        assert document.substage_runs[-1] == runner._FIELD_IDENTITY_SUBSTAGE
        calls.append("opinion")
        return document.complete_substage(runner._COURTLISTENER_OPINION_SUBSTAGE)

    def recap(document: Document, *, retrospective_date: date | None, client: object) -> Document:
        assert client is selected_client
        calls.append("recap")
        return document.complete_substage(runner._COURTLISTENER_RECAP_SUBSTAGE)

    def govinfo(document: Document, *, retrospective_date: date | None) -> Document:
        calls.append("govinfo")
        return document.complete_substage(runner._GOVINFO_OPINION_SUBSTAGE)

    async def review(document: Document) -> Document:
        calls.append("review")
        return document.complete_substage(runner._LOCATOR_BODY_REVIEW_SUBSTAGE)

    _patch_operation(monkeypatch, "retrieve_courtlistener_locator_opinions", opinion)
    _patch_operation(monkeypatch, "retrieve_courtlistener_locator_filings", recap)
    _patch_operation(monkeypatch, "retrieve_govinfo_locator_opinions", govinfo)
    _patch_operation(monkeypatch, "review_locator_bodies_llm", review)
    result = asyncio.run(
        workflow.validate_roots_citation_identity(
            document, courtlistener_client=selected_client, checkpoint=checkpoints.append
        )
    )
    assert result.substage_runs == runner._RUN_SUBSTAGES
    assert calls == ["opinion", "recap", "govinfo", "review"]
    remaining_substages = runner._RUN_SUBSTAGES[len(input_substages) :]
    expected_events = []
    for substage in remaining_substages:
        expected_events.append(("substage", substage))
        if substage == runner._FIELD_IDENTITY_SUBSTAGE:
            expected_events.append(("stage", "validate_roots_citation_identity.lookup_docket_roots"))
        elif substage == runner._LOCATOR_BODY_REVIEW_SUBSTAGE:
            expected_events.append(("stage", "validate_roots_citation_identity.corroborate_locator_bodies"))
    assert [
        (checkpoint.runs[-1].kind, checkpoint.runs[-1].name) for checkpoint in checkpoints
    ] == expected_events
    for checkpoint, (kind, name) in zip(checkpoints, expected_events, strict=True):
        recovered = result.get_stage(name) if kind == "stage" else result.get_substage(name)
        assert checkpoint == recovered
        assert checkpoint.runs == result.runs[: len(checkpoint.runs)]


def test_validation_replay_checks_all_sources_before_body_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filenames = ("001.txt", "002.txt")
    data_root = _dataset(tmp_path, filenames)
    checkpoint_dir = tmp_path / "validation-input"
    checkpoint_dir.mkdir()
    for filename in filenames:
        source = data_root / "primary" / "documents_txt" / filename
        ready = _complete(Document.from_source(source), runner._VALIDATION_INPUT_SUBSTAGES)
        (checkpoint_dir / f"{filename}.json").write_text(ready.model_dump_json(), encoding="utf-8")
    (data_root / "primary" / "documents_txt" / "002.txt").write_text("changed source\n", encoding="utf-8")

    async def unexpected_body(
        _document: Document,
        *,
        retrospective_date: date | None = None,
        checkpoint: Callable[[Document], None] | None = None,
    ) -> Document:
        pytest.fail("A stale validation checkpoint must not reach body providers")

    monkeypatch.setattr(_ROOT_WORKFLOW, "corroborate_locator_bodies", unexpected_body)
    with pytest.raises(ValueError, match="Saved Document text differs"):
        asyncio.run(
            runner._run(data_root, tmp_path / "results", None, from_validation_documents=checkpoint_dir)
        )


def test_transient_stage21_checkpoint_stops_later_work_and_retries_from_stage21(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    source_path = data_root / "primary" / "documents_txt" / "001.txt"
    source = "Acme v. Reed, Case No. 2:31-cv-45821 (S.D.N.Y. 2031)."
    source_path.write_text(source, encoding="utf-8")
    locator = "Case No. 2:31-cv-45821"
    number = "2:31-cv-45821"
    ready = Document.from_source(source_path).complete_substage(runner._RUN_SUBSTAGES[0])
    root = FullDocketCitation.from_locator(
        citation_id="docket:0",
        substage=runner._RUN_SUBSTAGES[1],
        source=source,
        span=Span(source.index(locator), source.index(locator) + len(locator)),
        number_span=Span(source.index(number), source.index(number) + len(number)),
    )
    ready = ready.add_citation(root).complete_substage(runner._RUN_SUBSTAGES[1])
    ready = _complete(ready, runner._RUN_SUBSTAGES[2:10])
    ready = ready.replace_citation(root.record(runner._ROOT_SUBSTAGE).with_root(root.id)).complete_substage(
        runner._ROOT_SUBSTAGE
    )
    ready = _complete(ready, runner._RUN_SUBSTAGES[11:_BODY_START])
    checkpoint_dir = tmp_path / "validation-input"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "001.txt.json").write_text(ready.model_dump_json(), encoding="utf-8")

    workflow = importlib.import_module("mellea_lrc.workflows.validate_roots_citation_identity")
    monkeypatch.setattr(workflow, "corroborate_locator_bodies", _LOCATOR_REVIEW)
    calls: list[str] = []
    cutoff = date(2024, 6, 1)

    def retrieve(substage: str, body_source: BodySource):
        def run(document: Document, *, retrospective_date: date | None = None) -> Document:
            assert retrospective_date == cutoff
            calls.append(substage)
            recorded = document.roots[0].record(substage)
            failure = (
                BodyEvidenceFailure(failure_type="transport_error", message="Temporary outage")
                if substage == runner._COURTLISTENER_RECAP_SUBSTAGE
                and calls.count(runner._COURTLISTENER_RECAP_SUBSTAGE) == 1
                else None
            )
            search = BodySearch(
                node_id=recorded.nodes[-1].id,
                source=body_source,
                retrospective_date=retrospective_date,
                attempts=(BodySearchAttempt(query=number, failure=failure),),
                failures=(failure,) if failure is not None else (),
            )
            return document.replace_citation(recorded.with_body_search(search)).complete_substage(substage)

        return run

    def govinfo(document: Document, *, retrospective_date: date | None = None) -> Document:
        assert retrospective_date == cutoff
        calls.append(runner._GOVINFO_OPINION_SUBSTAGE)
        return document.complete_substage(runner._GOVINFO_OPINION_SUBSTAGE)

    async def review(document: Document) -> Document:
        calls.append(runner._LOCATOR_BODY_REVIEW_SUBSTAGE)
        return document.complete_substage(runner._LOCATOR_BODY_REVIEW_SUBSTAGE)

    _patch_operation(
        monkeypatch,
        "retrieve_courtlistener_locator_opinions",
        retrieve(runner._COURTLISTENER_OPINION_SUBSTAGE, BodySource.COURTLISTENER_OPINION),
    )
    _patch_operation(
        monkeypatch,
        "retrieve_courtlistener_locator_filings",
        retrieve(runner._COURTLISTENER_RECAP_SUBSTAGE, BodySource.COURTLISTENER_RECAP),
    )
    _patch_operation(monkeypatch, "retrieve_govinfo_locator_opinions", govinfo)
    _patch_operation(monkeypatch, "review_locator_bodies_llm", review)

    with pytest.raises(RuntimeError, match=r"retrieve_courtlistener_filings.*transient provider failure"):
        asyncio.run(
            runner._run(
                data_root,
                tmp_path / "results",
                None,
                from_validation_documents=checkpoint_dir,
                retrospective_date=cutoff,
            )
        )
    run_dir = next((tmp_path / "results").iterdir())
    stage21 = Document.model_validate_json((run_dir / "documents" / "001.txt.json").read_text())
    assert stage21.substage_runs[-1] == runner._COURTLISTENER_RECAP_SUBSTAGE
    assert runner._GOVINFO_OPINION_SUBSTAGE not in stage21.substage_runs
    stage20 = stage21.get_substage(runner._COURTLISTENER_OPINION_SUBSTAGE)
    assert not (run_dir / "checkpoints").exists()
    assert runner._transient_body_failure_substage(stage21) == runner._COURTLISTENER_RECAP_SUBSTAGE
    assert calls == [runner._COURTLISTENER_OPINION_SUBSTAGE, runner._COURTLISTENER_RECAP_SUBSTAGE]

    assert asyncio.run(runner._run(tmp_path / "unused", tmp_path / "unused", None, run_dir)) == run_dir
    final = Document.model_validate_json((run_dir / "documents" / "001.txt.json").read_text())
    assert final.get_substage(runner._COURTLISTENER_OPINION_SUBSTAGE) == stage20
    assert not runner._has_transient_body_search_failure(final)
    assert calls == [
        runner._COURTLISTENER_OPINION_SUBSTAGE,
        runner._COURTLISTENER_RECAP_SUBSTAGE,
        runner._COURTLISTENER_RECAP_SUBSTAGE,
        runner._GOVINFO_OPINION_SUBSTAGE,
        runner._LOCATOR_BODY_REVIEW_SUBSTAGE,
    ]


@pytest.mark.parametrize(
    ("failed_substage", "failure_type", "status"),
    [
        (runner._COURTLISTENER_OPINION_SUBSTAGE, "http_error", 429),
        (runner._COURTLISTENER_RECAP_SUBSTAGE, "transport_error", None),
        (runner._GOVINFO_OPINION_SUBSTAGE, "http_error", 503),
    ],
)
@pytest.mark.parametrize("per_case", [False, True])
def test_transient_body_search_failure_replays_from_failed_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_substage: str,
    failure_type: str,
    status: int | None,
    per_case: bool,
) -> None:
    filenames = ("001.txt", "002.txt")
    data_root = _dataset(tmp_path, filenames)
    source = "Acme v. Reed, Case No. 2:31-cv-45821 (S.D.N.Y. 2031)."
    checkpoint_dir = tmp_path / "validation-input"
    checkpoint_dir.mkdir()
    locator = "Case No. 2:31-cv-45821"
    number = "2:31-cv-45821"
    for filename in filenames:
        source_path = data_root / "primary" / "documents_txt" / filename
        source_path.write_text(source, encoding="utf-8")
        ready = Document.from_source(source_path).complete_substage(runner._RUN_SUBSTAGES[0])
        root = FullDocketCitation.from_locator(
            citation_id="docket:0",
            substage=runner._RUN_SUBSTAGES[1],
            source=source,
            span=Span(source.index(locator), source.index(locator) + len(locator)),
            number_span=Span(source.index(number), source.index(number) + len(number)),
        )
        ready = ready.add_citation(root).complete_substage(runner._RUN_SUBSTAGES[1])
        ready = _complete(ready, runner._RUN_SUBSTAGES[2:10])
        ready = ready.replace_citation(
            root.record(runner._ROOT_SUBSTAGE).with_root(root.id)
        ).complete_substage(runner._ROOT_SUBSTAGE)
        ready = _complete(ready, runner._RUN_SUBSTAGES[11:_BODY_START])
        (checkpoint_dir / f"{filename}.json").write_text(ready.model_dump_json(), encoding="utf-8")

    cutoff = date(2024, 6, 1)
    expected_cutoffs = {"001.txt": cutoff, "002.txt": date(2025, 3, 4) if per_case else cutoff}
    if per_case:
        _annotation_cutoffs(data_root, {name: value.isoformat() for name, value in expected_cutoffs.items()})
    calls: list[tuple[str, str]] = []
    stages = (
        (runner._COURTLISTENER_OPINION_SUBSTAGE, BodySource.COURTLISTENER_OPINION),
        (runner._COURTLISTENER_RECAP_SUBSTAGE, BodySource.COURTLISTENER_RECAP),
        (runner._GOVINFO_OPINION_SUBSTAGE, BodySource.GOVINFO_OPINION),
    )

    def fake_provider(substage: str, body_source: BodySource):
        def run(document: Document, *, retrospective_date: date | None = None) -> Document:
            filename = Path(document.source_path or "").name
            assert retrospective_date == expected_cutoffs[filename]
            assert document.substage_runs == runner._RUN_SUBSTAGES[: runner._RUN_SUBSTAGES.index(substage)]
            first_call = (filename, substage) not in calls
            calls.append((filename, substage))
            root = document.roots[0].record(substage)
            failure = (
                BodyEvidenceFailure(
                    failure_type=failure_type,
                    message="Provider request failed",
                    status_code=status,
                )
                if filename == "001.txt" and substage == failed_substage and first_call
                else None
            )
            search = BodySearch(
                node_id=root.nodes[-1].id,
                source=body_source,
                retrospective_date=retrospective_date,
                attempts=(BodySearchAttempt(query=number, failure=failure),),
                failures=(failure,) if failure is not None else (),
            )
            return document.replace_citation(root.with_body_search(search)).complete_substage(substage)

        return run

    monkeypatch.setattr(runner, "retrieve_courtlistener_locator_opinions", fake_provider(*stages[0]))
    monkeypatch.setattr(runner, "retrieve_courtlistener_locator_filings", fake_provider(*stages[1]))
    monkeypatch.setattr(runner, "retrieve_govinfo_locator_opinions", fake_provider(*stages[2]))

    async def review(document: Document) -> Document:
        filename = Path(document.source_path or "").name
        assert document.substage_runs == runner._RUN_SUBSTAGES[:-1]
        calls.append((filename, runner._LOCATOR_BODY_REVIEW_SUBSTAGE))
        return document.complete_substage(runner._LOCATOR_BODY_REVIEW_SUBSTAGE)

    monkeypatch.setattr(runner, "review_locator_bodies_llm", review)

    async def body(
        document: Document,
        *,
        retrospective_date: date | None = None,
        courtlistener_client: object | None = None,
        checkpoint: Callable[[Document], None] | None = None,
    ) -> Document:
        assert document.substage_runs == _BODY_INPUT_SUBSTAGES
        for substage, _source in stages:
            if substage == runner._COURTLISTENER_OPINION_SUBSTAGE:
                document = runner.retrieve_courtlistener_locator_opinions(
                    document, retrospective_date=retrospective_date
                )
            elif substage == runner._COURTLISTENER_RECAP_SUBSTAGE:
                document = runner.retrieve_courtlistener_locator_filings(
                    document, retrospective_date=retrospective_date
                )
            else:
                document = runner.retrieve_govinfo_locator_opinions(
                    document, retrospective_date=retrospective_date
                )
        # This double models an older body run that saved only its final Document.
        # The artifact retry path must still work when no intermediate callback ran.
        document = await runner.review_locator_bodies_llm(document)
        return complete_stage_boundary(document, runner._FIELD_RUN_SUBSTAGES)

    monkeypatch.setattr(_ROOT_WORKFLOW, "corroborate_locator_bodies", body)
    with pytest.raises(RuntimeError, match="transient provider failures"):
        asyncio.run(
            runner._run(
                data_root,
                tmp_path / "results",
                None,
                from_validation_documents=checkpoint_dir,
                retrospective_date=None if per_case else cutoff,
                annotation_case_cutoffs=per_case,
            )
        )
    run_dir = next((tmp_path / "results").iterdir())
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["transient_failures"] == {"docket": [], "body": ["001.txt"]}
    initial = Document.model_validate_json(
        (run_dir / "documents" / "001.txt.json").read_text(encoding="utf-8")
    )
    assert runner._transient_body_failure_substage(initial) == failed_substage
    previous_substage = runner._RUN_SUBSTAGES[runner._RUN_SUBSTAGES.index(failed_substage) - 1]
    first_calls = len(calls)

    assert asyncio.run(runner._run(tmp_path / "unused", tmp_path / "unused", None, run_dir)) == run_dir
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "complete"
    assert "transient_failures" not in record
    assert calls[first_calls:] == [
        ("001.txt", substage)
        for substage in (
            *runner._BODY_SEARCH_SUBSTAGES[runner._BODY_SEARCH_SUBSTAGES.index(failed_substage) :],
            runner._LOCATOR_BODY_REVIEW_SUBSTAGE,
        )
    ]
    final = Document.model_validate_json((run_dir / "documents" / "001.txt.json").read_text(encoding="utf-8"))
    assert final.get_substage(previous_substage) == initial.get_substage(previous_substage)
    assert not runner._has_transient_body_search_failure(final)


@pytest.mark.parametrize(
    ("failure_type", "status"),
    [("http_error", 429), ("transport_error", None)],
)
def test_transient_docket_search_failure_requires_a_rerun(failure_type: str, status: int | None) -> None:
    document = asyncio.run(grow_roots(Document.from_source("Acme v. Reed, Case No. 2:31-cv-45821.")))
    root = document.roots[0]
    assert isinstance(root, FullDocketCitation)
    recorded = root.record(
        "validate_roots_citation_identity.lookup_docket_roots.retrieve_courtlistener_dockets"
    )
    lookup = DocketLookup(
        node_id=recorded.nodes[-1].id,
        attempts=(
            DocketLookupAttempt(
                source_type="d",
                query="docketNumber:(2:31-cv-45821)",
                failure=DocketLookupFailure(
                    failure_type=failure_type,
                    message="Provider request failed",
                    upstream_status_code=status,
                ),
            ),
        ),
    )
    incomplete = document.replace_citation(recorded.with_docket_lookup(lookup))

    assert runner._has_transient_docket_lookup_failure(incomplete)
    assert not runner._has_transient_docket_lookup_failure(document)


def test_transient_govinfo_search_failure_requires_a_rerun() -> None:
    document = asyncio.run(grow_roots(Document.from_source("Acme v. Reed, Case No. 2:31-cv-45821.")))
    root = document.roots[0]
    assert isinstance(root, FullDocketCitation)
    recorded = root.record("validate_roots_citation_identity.lookup_docket_roots.retrieve_govinfo_dockets")
    lookup = GovInfoDocketLookup(
        node_id=recorded.nodes[-1].id,
        attempts=(
            GovInfoLookupAttempt(
                query='collection:uscourts casenumber:("2:31-cv-45821")',
                failure=DocketLookupFailure(
                    failure_type="http_error",
                    message="Provider quota reached",
                    upstream_status_code=429,
                ),
            ),
        ),
    )
    incomplete = document.replace_citation(recorded.with_govinfo_docket_lookup(lookup))
    assert runner._has_transient_docket_lookup_failure(incomplete)


def test_new_run_stays_failed_when_a_saved_search_is_transiently_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))

    async def fake_grow(
        document: Document,
        *,
        hunt_dockets: bool,
        review_docket_roots: bool,
        field_extraction,
        checkpoint=None,
    ) -> Document:
        return _complete(document, runner._RUN_SUBSTAGES[:12])

    async def fake_validate(
        document: Document,
        *,
        retrospective_date: date | None = None,
        checkpoint: Callable[[Document], None] | None = None,
    ) -> Document:
        document = _complete_with_checkpoints(document, runner._RUN_SUBSTAGES[12:_BODY_START], checkpoint)
        return _complete_with_checkpoints(document, _BODY_SUBSTAGES, checkpoint)

    monkeypatch.setattr(runner, "grow_roots", fake_grow)
    monkeypatch.setattr(runner, "validate_roots_citation_identity", fake_validate)
    monkeypatch.setattr(runner, "_has_transient_docket_lookup_failure", lambda _document: True)

    with pytest.raises(RuntimeError, match="transient provider failures"):
        asyncio.run(runner._run(data_root, tmp_path / "results", None))

    run_dir = next((tmp_path / "results").iterdir())
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert (run_dir / "documents" / "001.txt.json").exists()


def _docket_roots_input(source_path: Path) -> Document:
    document = Document.from_source(source_path).complete_substage(runner._RUN_SUBSTAGES[0])
    locator, number = "Case No. 2:31-cv-45821", "2:31-cv-45821"
    root = FullDocketCitation.from_locator(
        citation_id="docket:0",
        substage=runner._RUN_SUBSTAGES[1],
        source=document.text,
        span=Span(document.text.index(locator), document.text.index(locator) + len(locator)),
        number_span=Span(document.text.index(number), document.text.index(number) + len(number)),
    )
    document = document.add_citation(root).complete_substage(runner._RUN_SUBSTAGES[1])
    document = _complete(document, runner._RUN_SUBSTAGES[2:10])
    document = document.replace_citation(
        root.record(runner._ROOT_SUBSTAGE).with_root(root.id)
    ).complete_substage(runner._ROOT_SUBSTAGE)
    return _complete(document, runner._RUN_SUBSTAGES[11:12])


def _docket_failure(document: Document, substage: str, *, failure_type="transport_error", status=None):
    root = document.roots[0].record(substage)
    failure = DocketLookupFailure(
        failure_type=failure_type, message="Offline transient provider failure", upstream_status_code=status
    )
    if substage == runner._DOCKET_LOOKUP_SUBSTAGE:
        lookup = DocketLookup(
            node_id=root.nodes[-1].id,
            attempts=(DocketLookupAttempt(source_type="d", query="offline", failure=failure),),
        )
        root = root.with_docket_lookup(lookup)
    else:
        lookup = GovInfoDocketLookup(
            node_id=root.nodes[-1].id,
            attempts=(GovInfoLookupAttempt(query="offline", failure=failure),),
        )
        root = root.with_govinfo_docket_lookup(lookup)
    return document.replace_citation(root)


@pytest.mark.parametrize("input_mode", ["source", "saved_roots"])
@pytest.mark.parametrize(
    ("failed_substage", "failure_type", "status"),
    [
        (runner._DOCKET_LOOKUP_SUBSTAGE, "transport_error", None),
        (runner._DOCKET_LOOKUP_SUBSTAGE, "http_error", 429),
        ("validate_roots_citation_identity.lookup_docket_roots.retrieve_govinfo_dockets", "http_error", 503),
    ],
)
def test_ordinary_resume_retries_only_the_failed_docket_provider(
    tmp_path, monkeypatch, input_mode, failed_substage, failure_type, status
):
    data_root = _dataset(tmp_path, ("001.txt",))
    source_path = data_root / "primary" / "documents_txt" / "001.txt"
    source_path.write_text("Acme v. Reed, Case No. 2:31-cv-45821.")
    roots = _docket_roots_input(source_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "001.txt.json").write_text(roots.model_dump_json())
    root_calls = []
    calls = []
    interrupted = False

    async def grow(document, **_kwargs):
        root_calls.append(document)
        return roots

    async def validate(document, *, checkpoint, **_kwargs):
        nonlocal interrupted
        for substage in runner._RUN_SUBSTAGES[12:]:
            if substage in document.substage_runs:
                continue
            calls.append(substage)
            if substage == failed_substage and not interrupted:
                interrupted = True
                document = _docket_failure(document, substage, failure_type=failure_type, status=status)
            document = _complete_with_checkpoints(document, (substage,), checkpoint)
        return document

    monkeypatch.setattr(runner, "grow_roots", grow)
    monkeypatch.setattr(runner, "validate_roots_citation_identity", validate)
    with pytest.raises(RuntimeError, match="transient provider failure"):
        asyncio.run(
            runner._run(data_root, tmp_path / "results", inputs if input_mode == "saved_roots" else None)
        )
    run_dir = next((tmp_path / "results").iterdir())
    artifact = run_dir / "documents" / "001.txt.json"
    failed = Document.model_validate_json(artifact.read_text())
    previous = runner._RUN_SUBSTAGES[runner._RUN_SUBSTAGES.index(failed_substage) - 1]
    assert failed.substage_runs[-1] == failed_substage
    first_calls = list(calls)

    assert asyncio.run(runner._run(data_root, tmp_path / "results", None, resume_run=run_dir)) == run_dir
    completed = Document.model_validate_json(artifact.read_text())
    assert completed.get_substage(previous) == failed.get_substage(previous)
    assert calls[len(first_calls) :] == list(
        runner._RUN_SUBSTAGES[runner._RUN_SUBSTAGES.index(failed_substage) :]
    )
    assert len(root_calls) == 1
    assert not runner._has_transient_docket_lookup_failure(completed)
    assert json.loads((run_dir / "run.json").read_text())["status"] == "complete"


@pytest.mark.parametrize("input_mode", ["checkpoint", "validation", "reporter_reuse"])
def test_preserved_input_with_a_failed_docket_lookup_is_rejected_before_provider_work(
    tmp_path, monkeypatch, input_mode
):
    data_root = _dataset(tmp_path, ("001.txt", "002.txt"))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for filename in ("001.txt", "002.txt"):
        source_path = data_root / "primary" / "documents_txt" / filename
        source_path.write_text("Acme v. Reed, Case No. 2:31-cv-45821.")
        document = _docket_roots_input(source_path)
        for substage in runner._RUN_SUBSTAGES[
            12 : runner._RUN_SUBSTAGES.index(runner._VALIDATION_INPUT_SUBSTAGE) + 1
        ]:
            if filename == "002.txt" and substage == runner._DOCKET_LOOKUP_SUBSTAGE:
                document = _docket_failure(document, substage)
            document = _complete(document, (substage,))
        (inputs / f"{filename}.json").write_text(document.model_dump_json())

    def unexpected(*_args, **_kwargs):
        pytest.fail("A failed immutable input must be rejected before provider or workflow calls")

    for name in ("grow_roots", "validate_roots_citation_identity", "CourtListenerClient"):
        monkeypatch.setattr(runner, name, unexpected)
    options = {
        "checkpoint": {
            "from_checkpoint_documents": inputs,
            "checkpoint_substage": runner._DOCKET_LOOKUP_SUBSTAGE,
        },
        "validation": {"from_validation_documents": inputs},
        "reporter_reuse": {"from_reporter_review_documents": inputs, "reuse_docket_lookups": True},
    }[input_mode]
    with pytest.raises(ValueError, match="transient failure"):
        asyncio.run(runner._run(data_root, tmp_path / "results", None, **options))
    assert list((tmp_path / "results").rglob("documents/*.json")) == []
    assert json.loads(next((tmp_path / "results").glob("*/run.json")).read_text())["status"] == "failed"


def test_preserved_body_failure_is_rejected_before_replaying_its_input(tmp_path, monkeypatch):
    data_root = _dataset(tmp_path, ("001.txt",))
    source_path = data_root / "primary" / "documents_txt" / "001.txt"
    source_path.write_text("Acme v. Reed, Case No. 2:31-cv-45821.")
    document = _docket_roots_input(source_path)
    substage = runner._COURTLISTENER_OPINION_SUBSTAGE
    document = _complete(document, runner._RUN_SUBSTAGES[12 : runner._RUN_SUBSTAGES.index(substage)])
    root = document.roots[0].record(substage)
    failure = BodyEvidenceFailure(failure_type="transport_error", message="Offline transient failure")
    search = BodySearch(
        node_id=root.nodes[-1].id,
        source=BodySource.COURTLISTENER_OPINION,
        retrospective_date=None,
        attempts=(BodySearchAttempt(query="offline", failure=failure),),
        failures=(failure,),
    )
    document = document.replace_citation(root.with_body_search(search)).complete_substage(substage)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "001.txt.json").write_text(document.model_dump_json())

    def unexpected(*_args, **_kwargs):
        pytest.fail("An immutable failed body retrieval must not execute provider or workflow calls")

    monkeypatch.setattr(runner, "validate_roots_citation_identity", unexpected)
    monkeypatch.setattr(runner, "CourtListenerClient", unexpected)
    with pytest.raises(ValueError, match="choose an input checkpoint before that retrieval"):
        asyncio.run(
            runner._run(
                data_root,
                tmp_path / "results",
                None,
                from_checkpoint_documents=inputs,
                checkpoint_substage=substage,
            )
        )
    assert list((tmp_path / "results").rglob("documents/*.json")) == []


def test_resume_repairs_missing_final_group_after_all_substages_without_provider_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    source = Document.from_source(data_root / "primary" / "documents_txt" / "001.txt")
    completed = _complete(source, runner._RUN_SUBSTAGES)
    final_stage = "validate_roots_citation_identity.corroborate_locator_bodies"
    ungrouped = completed.get_substage(runner._LOCATOR_BODY_REVIEW_SUBSTAGE)
    assert ungrouped.substage_runs == runner._RUN_SUBSTAGES
    assert final_stage not in ungrouped.stage_runs
    run_dir = tmp_path / "saved-run"
    artifacts = run_dir / "documents"
    artifacts.mkdir(parents=True)
    artifact = artifacts / "001.txt.json"
    artifact.write_text(ungrouped.model_dump_json(), encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "set": "primary",
                "status": "failed",
                "filings": ["001.txt"],
                "data_root": str(data_root),
                "rendering_sha256": {"001.txt": hashlib.sha256(b"{}").hexdigest()},
                "field_extraction": _RULE_FIELDS.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )

    def reject_provider(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Repairing a completed group must not execute any provider or workflow")

    for name in (
        "grow_roots",
        "validate_roots_citation_identity",
        "CourtListenerClient",
        "retrieve_courtlistener_docket_roots",
        "review_courtlistener_docket_roots_llm",
        "retrieve_govinfo_docket_roots",
        "review_govinfo_docket_roots_llm",
        "review_unique_reporter_roots_llm",
        "review_ambiguous_reporter_roots_llm",
        "retrieve_courtlistener_locator_opinions",
        "retrieve_courtlistener_locator_filings",
        "retrieve_govinfo_locator_opinions",
        "review_locator_bodies_llm",
    ):
        monkeypatch.setattr(runner, name, reject_provider)
    assert asyncio.run(runner._run(data_root, tmp_path / "unused", None, resume_run=run_dir)) == run_dir
    restored = Document.model_validate_json(artifact.read_text(encoding="utf-8"))
    assert restored.model_dump_json() == completed.model_dump_json()
    assert (
        restored.get_stage(final_stage).model_dump_json()
        == completed.get_stage(final_stage).model_dump_json()
    )
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_root_extraction_resume_keeps_saved_prefix_and_persists_before_review_failure(tmp_path, monkeypatch):
    import inspect

    from mellea_lrc.model.execution import get_workflow

    data_root = _dataset(tmp_path, ("001.txt",))
    source = Document.from_source(data_root / "primary" / "documents_txt" / "001.txt")
    original = _complete(source, runner._ROOT_SUBSTAGES[:2])
    run_dir = tmp_path / "saved-run"
    artifacts = run_dir / "documents"
    artifacts.mkdir(parents=True)
    artifact = artifacts / "001.txt.json"
    artifact.write_text(original.model_dump_json())
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "set": "primary",
                "status": "failed",
                "filings": ["001.txt"],
                "data_root": str(data_root),
                "rendering_sha256": {"001.txt": hashlib.sha256(b"{}").hexdigest()},
                "field_extraction": _RULE_FIELDS.model_dump(mode="json"),
            }
        )
    )
    calls = []
    fail_review = True

    def finish(document, substage):
        if substage in original.substage_runs:
            pytest.fail("A completed root extraction operation must not replay")
        calls.append(substage)
        if substage == "grow_roots.form_root_groups.reassign_docket_roots_llm" and fail_review:
            raise RuntimeError("offline interrupted review")
        return document.complete_substage(substage)

    def fake(function, substage):
        if inspect.iscoroutinefunction(function):

            async def run(document, *args, **kwargs):
                return finish(document, substage)
        else:

            def run(document, *args, **kwargs):
                return finish(document, substage)

        return run

    for stage in get_workflow("grow_roots").stages:
        module = importlib.import_module(f"mellea_lrc.workflows.{stage.name}")
        for name, function in vars(module).copy().items():
            if inspect.isfunction(function) and function.__module__.startswith("mellea_lrc.extraction."):
                substage = getattr(importlib.import_module(function.__module__), "SUBSTAGE", None)
                if substage is not None:
                    monkeypatch.setattr(module, name, fake(function, substage))

    async def validate(document, *, checkpoint=None, **kwargs):
        return _complete_with_checkpoints(
            document, runner._RUN_SUBSTAGES[len(runner._ROOT_SUBSTAGES) + 1 :], checkpoint
        )

    monkeypatch.setattr(runner, "validate_roots_citation_identity", validate)
    with pytest.raises(RuntimeError, match="offline interrupted review"):
        asyncio.run(runner._run(data_root, tmp_path / "unused", None, resume_run=run_dir))
    saved = Document.model_validate_json(artifact.read_text())
    assert saved.substage_runs == runner._ROOT_SUBSTAGES
    assert saved.get_substage(original.substage_runs[-1]) == original
    assert saved.stage_runs == ("grow_roots.discover_root_locators", "grow_roots.read_root_fields")
    fail_review = False
    calls.clear()
    assert asyncio.run(runner._run(data_root, tmp_path / "unused", None, resume_run=run_dir)) == run_dir
    assert calls == ["grow_roots.form_root_groups.reassign_docket_roots_llm"]
    completed = Document.model_validate_json(artifact.read_text())
    assert completed.substage_runs == runner._RUN_SUBSTAGES
    assert completed.get_substage(original.substage_runs[-1]) == original
    assert "grow_roots.form_root_groups" in completed.stage_runs


def test_invalid_resume_prefix_does_not_rewrite_document_to_repair_its_final_group(tmp_path, monkeypatch):
    data_root = _dataset(tmp_path, ("001.txt",))
    source = Document.from_source(data_root / "primary" / "documents_txt" / "001.txt")
    source = source.complete_substage("development.invalid_checkpoint")
    completed = _complete(source, runner._RUN_SUBSTAGES)
    invalid = completed.get_substage(runner._LOCATOR_BODY_REVIEW_SUBSTAGE)
    assert "validate_roots_citation_identity.corroborate_locator_bodies" not in invalid.stage_runs
    run_dir = tmp_path / "saved-run"
    artifacts = run_dir / "documents"
    artifacts.mkdir(parents=True)
    artifact = artifacts / "001.txt.json"
    artifact.write_text(invalid.model_dump_json())
    before = artifact.read_bytes()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "set": "primary",
                "status": "failed",
                "filings": ["001.txt"],
                "data_root": str(data_root),
                "rendering_sha256": {"001.txt": hashlib.sha256(b"{}").hexdigest()},
                "field_extraction": _RULE_FIELDS.model_dump(mode="json"),
            }
        )
    )

    async def unexpected(*_args, **_kwargs):
        pytest.fail("An invalid checkpoint must not run a workflow")

    monkeypatch.setattr(runner, "grow_roots", unexpected)
    monkeypatch.setattr(runner, "validate_roots_citation_identity", unexpected)
    with pytest.raises(ValueError, match="Saved evaluation checkpoint skips a substage"):
        asyncio.run(runner._run(data_root, tmp_path / "unused", None, resume_run=run_dir))
    assert artifact.read_bytes() == before


def test_enabled_model_runtime_failure_happens_before_source_loading_or_artifact_writes(
    tmp_path, monkeypatch
):
    config = FieldExtractionConfig.from_env(environ=_ml_field_settings())
    monkeypatch.setattr(FieldExtractionConfig, "from_env", classmethod(lambda cls: config))

    def unavailable(_self):
        raise RuntimeError("Configured checkpoint unavailable")

    def unexpected(*_args, **_kwargs):
        pytest.fail("Runtime preflight must happen before loading sources or writing artifacts")

    monkeypatch.setattr(FieldExtractionConfig, "validate_runtime", unavailable)
    monkeypatch.setattr(runner, "load_source", unexpected)
    monkeypatch.setattr(runner, "_write_json", unexpected)
    with pytest.raises(RuntimeError, match="Configured checkpoint unavailable"):
        asyncio.run(runner._run(tmp_path / "missing-data", tmp_path / "results", None))
    assert not (tmp_path / "results").exists()
