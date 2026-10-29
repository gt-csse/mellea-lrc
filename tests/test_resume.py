"""Public resume reuses workflow boundaries without changing native Documents."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.model.document import Document
from mellea_lrc.model.execution import WORKFLOW_CATALOG


@pytest.fixture
def execution(monkeypatch):
    from mellea_lrc import execution

    fields = FieldExtractionConfig(
        ml_enabled=False, case_name="RULE", court="RULE", date="RULE", pin_cite="RULE"
    )
    monkeypatch.setattr(FieldExtractionConfig, "from_env", classmethod(lambda cls: fields))
    monkeypatch.setattr(execution, "_configuration", lambda fields: {"field_policy": "RULE"})
    monkeypatch.setattr(execution, "_implementation_fingerprint", lambda: {"test": "same-source"})
    return execution


@pytest.fixture
def source_and_output(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("A filing with no citations.", encoding="utf-8")
    return source, tmp_path / "document.json"


def test_configuration_records_settings_without_credential_values(monkeypatch):
    from mellea_lrc import execution
    from mellea_lrc.llm.config import LlmOutputMode
    from mellea_lrc.llm.profiles import LlmProfile
    from mellea_lrc.providers.courtlistener import CourtListenerConfig
    from mellea_lrc.providers.govinfo import GovInfoConfig

    fields = FieldExtractionConfig(
        ml_enabled=False, case_name="RULE", court="RULE", date="RULE", pin_cite="RULE"
    )
    profile = LlmProfile(
        name="test",
        model="test-model",
        api_base="https://models.example/v1",
        api_key_env="TEST_KEY",
        temperature=0,
        timeout_seconds=300,
        max_tokens=8000,
        max_attempts=3,
        output_mode=LlmOutputMode.JSON_SCHEMA,
    )
    credential = ["private-credential-one"]
    monkeypatch.setattr(execution, "load_profile", lambda name: profile)
    monkeypatch.setattr(LlmProfile, "resolve", lambda self: {"api_key": credential[0]})
    monkeypatch.setattr(
        CourtListenerConfig,
        "from_env",
        classmethod(
            lambda cls: cls(base_url="https://courts.example", timeout_seconds=45, token=credential[0])
        ),
    )
    monkeypatch.setattr(
        GovInfoConfig,
        "from_env",
        classmethod(
            lambda cls: cls(base_url="https://govinfo.example", api_key=credential[0], timeout_seconds=45)
        ),
    )
    first = execution._configuration(fields)
    assert credential[0] not in json.dumps(first)
    credential[0] = "private-credential-two"
    assert execution._configuration(fields) == first


def _fake_atomic_operations(monkeypatch, calls, *, interrupt_at=None):
    """Keep all four real workflow composers, replacing only their atomic work."""
    interrupted = False

    def apply(document, substage):
        nonlocal interrupted
        calls.append(substage)
        if substage == interrupt_at and not interrupted:
            interrupted = True
            raise RuntimeError("Synthetic provider interruption")
        return document.complete_substage(substage)

    def synchronous(substage):
        def run(document, *args, **kwargs):
            return apply(document, substage)

        return run

    def asynchronous(substage):
        async def run(document, *args, **kwargs):
            return apply(document, substage)

        return run

    for workflow in WORKFLOW_CATALOG:
        for stage in workflow.stages:
            module = importlib.import_module(f"mellea_lrc.workflows.{stage.name}")
            for name, function in vars(module).copy().items():
                if not inspect.isfunction(function) or not function.__module__.startswith(
                    ("mellea_lrc.extraction.", "mellea_lrc.validation.")
                ):
                    continue
                atomic = importlib.import_module(function.__module__)
                substage = getattr(atomic, "SUBSTAGE", None)
                if substage is None:
                    continue
                factory = asynchronous if inspect.iscoroutinefunction(function) else synchronous
                monkeypatch.setattr(module, name, factory(substage))


def _assert_no_work(monkeypatch, execution):
    async def unexpected(*args, **kwargs):
        pytest.fail("Resume validation failure reached a workflow")

    for workflow in WORKFLOW_CATALOG:
        monkeypatch.setattr(execution, workflow.name, unexpected)


def test_resume_uses_real_workflow_apis_and_skips_committed_atomic_work(
    execution, source_and_output, monkeypatch
):
    source, output = source_and_output
    calls = []
    interrupted = "validate_roots_citation_identity.lookup_reporter_roots.review_unique_llm"
    _fake_atomic_operations(monkeypatch, calls, interrupt_at=interrupted)
    with pytest.raises(RuntimeError, match="Synthetic provider interruption"):
        asyncio.run(execution.validate(source, output))

    saved = Document.model_validate_json(output.read_text())
    committed = set(saved.substage_runs)
    assert committed
    assert interrupted not in committed
    assert all(name.startswith("grow_roots.") for name in saved.stage_runs[:3])
    assert execution._state_dir(output) == output.with_name("document.json.run")

    restored = asyncio.run(execution.resume(output))
    assert isinstance(restored, Document)
    assert restored == Document.model_validate_json(output.read_text())
    counts = Counter(calls)
    assert all(counts[name] == 1 for name in committed)
    assert counts[interrupted] == 2  # Its first invocation never completed.
    assert "validate_all_citation_content.review_citation_content" in restored.stage_runs
    assert json.loads((execution._state_dir(output) / "run.json").read_text())["status"] == "complete"


def test_resuming_completed_run_does_no_work_or_rewrite_native_document(
    execution, source_and_output, monkeypatch
):
    source, output = source_and_output
    calls = []
    _fake_atomic_operations(monkeypatch, calls)
    completed = asyncio.run(execution.validate(source, output))
    before = output.read_bytes()
    _assert_no_work(monkeypatch, execution)

    assert asyncio.run(execution.resume(output)) == completed
    assert output.read_bytes() == before


@pytest.mark.parametrize("existing", ["document", "state_directory"])
def test_validate_never_overwrites_existing_run(execution, source_and_output, monkeypatch, existing):
    source, output = source_and_output
    original = Document.from_source("An existing native artifact.").model_dump_json()
    if existing == "document":
        output.write_text(original, encoding="utf-8")
    else:
        execution._state_dir(output).mkdir()

    def unexpected(*args, **kwargs):
        pytest.fail("Existing run reached preprocessing")

    monkeypatch.setattr(Document, "from_source", unexpected)
    _assert_no_work(monkeypatch, execution)
    with pytest.raises(FileExistsError):
        asyncio.run(execution.validate(source, output))
    if existing == "document":
        assert output.read_text() == original
    else:
        assert not output.exists()


def test_missing_resume_metadata_does_not_invalidate_old_document(execution, source_and_output, monkeypatch):
    _, output = source_and_output
    original = Document.from_source("Existing serialized Document.")
    output.write_text(original.model_dump_json(), encoding="utf-8")
    _assert_no_work(monkeypatch, execution)

    with pytest.raises(ValueError, match=r"(?i)(metadata|state|run|resume)"):
        asyncio.run(execution.resume(output))
    assert Document.model_validate_json(output.read_text()) == original


def test_interrupted_preprocessing_can_start_again_without_orphaned_run_state(
    execution, source_and_output, monkeypatch
):
    source, output = source_and_output
    original = Document.from_source
    attempts = []

    def preprocess(cls, path):
        attempts.append(path)
        if len(attempts) == 1:
            raise KeyboardInterrupt()
        return original(path)

    monkeypatch.setattr(Document, "from_source", classmethod(preprocess))
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(execution.validate(source, output))
    assert not output.exists()
    assert not execution._state_dir(output).exists()
    calls = []
    _fake_atomic_operations(monkeypatch, calls)
    asyncio.run(execution.validate(source, output))
    assert len(attempts) == 2
    assert json.loads((execution._state_dir(output) / "run.json").read_text())["status"] == "complete"


def test_source_change_during_preprocessing_fails_before_initial_checkpoint(
    execution, source_and_output, monkeypatch
):
    source, output = source_and_output
    original = Document.from_source

    def preprocess(cls, path):
        document = original(path)
        path.write_text("Different source while preprocessing")
        return document

    monkeypatch.setattr(Document, "from_source", classmethod(preprocess))
    with pytest.raises(ValueError, match="changed during preprocessing"):
        asyncio.run(execution.validate(source, output))
    assert not output.exists()
    assert not execution._state_dir(output).exists()


@pytest.mark.parametrize("change", ["configuration", "implementation", "source", "text", "metadata"])
def test_resume_rejects_incompatible_state_before_any_workflow(
    execution, source_and_output, monkeypatch, change
):
    source, output = source_and_output
    calls = []
    interrupted = "grow_roots.discover_root_locators.hunt_docket_locators"
    _fake_atomic_operations(monkeypatch, calls, interrupt_at=interrupted)
    with pytest.raises(RuntimeError, match="Synthetic provider interruption"):
        asyncio.run(execution.validate(source, output))
    if change == "configuration":
        monkeypatch.setattr(execution, "_configuration", lambda fields: {"field_policy": "ML"})
    elif change == "implementation":
        monkeypatch.setattr(execution, "_implementation_fingerprint", lambda: {"test": "changed-source"})
    elif change == "source":
        source.write_text("Changed original filing.", encoding="utf-8")
    else:
        saved = json.loads(output.read_text())
        if change == "text":
            saved["text"] += " Altered preprocessed text."
        else:
            saved["source_metadata"]["path"] = str(source.with_name("different.txt"))
        output.write_text(json.dumps(saved), encoding="utf-8")
    unchanged = output.read_bytes()
    _assert_no_work(monkeypatch, execution)

    with pytest.raises(ValueError):
        asyncio.run(execution.resume(output))
    assert output.read_bytes() == unchanged


def test_newer_atomic_document_resumes_when_partial_manifest_was_not_updated(
    execution, source_and_output, monkeypatch
):
    source, output = source_and_output
    calls = []
    interrupted = "grow_roots.discover_root_locators.hunt_docket_locators"
    _fake_atomic_operations(monkeypatch, calls, interrupt_at=interrupted)
    with pytest.raises(RuntimeError, match="Synthetic provider interruption"):
        asyncio.run(execution.validate(source, output))
    state = execution._state_dir(output) / "run.json"
    prior_manifest = state.read_bytes()
    document = Document.model_validate_json(output.read_text())
    # Simulate a fully saved next Document before an unrelated status write.
    document = document.complete_substage(interrupted)
    output.write_text(document.model_dump_json(), encoding="utf-8")
    assert state.read_bytes() == prior_manifest
    calls.clear()

    final = asyncio.run(execution.resume(output))
    assert interrupted not in calls
    assert final.get_substage(interrupted) == document
    assert "validate_all_citation_content.review_citation_content" in final.stage_runs


def test_completed_artifact_edit_is_rejected_before_workflow(execution, source_and_output, monkeypatch):
    source, output = source_and_output
    _fake_atomic_operations(monkeypatch, [])
    asyncio.run(execution.validate(source, output))
    # Whitespace is harmless to Document loading but changes the frozen artifact.
    output.write_bytes(output.read_bytes() + b"\n")
    _assert_no_work(monkeypatch, execution)
    with pytest.raises(ValueError):
        asyncio.run(execution.resume(output))


def test_active_run_lock_rejects_second_owner_without_changing_artifact(
    execution, source_and_output, monkeypatch
):
    import fcntl

    source, output = source_and_output
    interrupted = "grow_roots.discover_root_locators.hunt_docket_locators"
    _fake_atomic_operations(monkeypatch, [], interrupt_at=interrupted)
    with pytest.raises(RuntimeError, match="Synthetic provider interruption"):
        asyncio.run(execution.validate(source, output))
    prior_document = output.read_bytes()
    prior_manifest = (execution._state_dir(output) / "run.json").read_bytes()
    _assert_no_work(monkeypatch, execution)

    with (execution._state_dir(output) / "owner.lock").open("a+") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Run is already active"):
            asyncio.run(execution.resume(output))
    assert output.read_bytes() == prior_document
    assert (execution._state_dir(output) / "run.json").read_bytes() == prior_manifest


def test_failed_atomic_checkpoint_write_keeps_last_complete_document(
    execution, source_and_output, monkeypatch
):
    source, output = source_and_output
    calls = []
    _fake_atomic_operations(monkeypatch, calls)
    replace = Path.replace

    def fail_second_atomic_write(path, target):
        if Path(target) == output:
            candidate = Document.model_validate_json(path.read_text())
            if len(candidate.substage_runs) == 2:
                raise OSError("Cannot replace next checkpoint")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_second_atomic_write)
    with pytest.raises(OSError, match="Cannot replace next checkpoint"):
        asyncio.run(execution.validate(source, output))
    saved = Document.model_validate_json(output.read_text())
    assert saved.substage_runs == ("grow_roots.discover_root_locators.find_full_reporter_locators",)
    assert json.loads((execution._state_dir(output) / "run.json").read_text())["status"] == "interrupted"


def test_resume_reuses_completed_success_and_failed_reviews_inside_unfinished_substage(
    execution, source_and_output, monkeypatch
):
    from mellea.stdlib.sampling import MultiTurnStrategy

    from mellea_lrc.llm import ivr
    from mellea_lrc.llm.journal import active_journal
    from mellea_lrc.model.ivr import IvrAttempt, IvrRequirementAttempt, IvrRun

    source, output = source_and_output
    _fake_atomic_operations(monkeypatch, [])
    session = SimpleNamespace(backend=SimpleNamespace(model_id="test-model"))
    strategy = MultiTurnStrategy(loop_budget=3)
    native_calls = []
    first_results = []
    resumed_results = []
    invocations = 0
    reused = []

    async def native(session, spec, *, strategy, model_options):
        identifier = spec.user_variables["citation"]
        native_calls.append(identifier)
        success = identifier != "two"
        return IvrRun(
            success=success,
            selected_attempt=0,
            attempts=(
                IvrAttempt(
                    output=json.dumps({"citation": identifier}),
                    requirements=(
                        IvrRequirementAttempt(
                            description="The evidence quote must ground in the source.",
                            passed=success,
                            reason=None if success else "The quote did not ground.",
                            score=None,
                        ),
                    ),
                ),
            ),
            **ivr._run_metadata(session, spec, model_options),
        )

    async def review_sites(document, **kwargs):
        nonlocal invocations
        invocations += 1
        journal = active_journal()
        assert journal is not None
        observed = []
        for identifier in ("one", "two"):
            result = await ivr.run_instruct_ivr(
                session,
                ivr.InstructIvrSpec(
                    description="Review this citation against the supplied source.",
                    grounding_context={"source": document.text},
                    user_variables={"citation": identifier},
                ),
                strategy=strategy,
                model_options={},
            )
            observed.append(result.model_dump(mode="json"))
        if invocations == 1:
            first_results.extend(observed)
            raise RuntimeError("Process interrupted before the substage checkpoint")
        resumed_results.extend(observed)
        reused.append(journal.reused)
        await ivr.run_instruct_ivr(
            session,
            ivr.InstructIvrSpec(
                description="Review this citation against the supplied source.",
                grounding_context={"source": document.text},
                user_variables={"citation": "three"},
            ),
            strategy=strategy,
            model_options={},
        )
        return document.complete_substage("grow_roots.discover_root_locators.hunt_docket_locators")

    monkeypatch.setattr(ivr, "_run_instruct_ivr", native)
    stage = importlib.import_module("mellea_lrc.workflows.grow_roots.discover_root_locators")
    monkeypatch.setattr(stage, "hunt_docket_locators", review_sites)
    with pytest.raises(RuntimeError, match="Process interrupted before the substage checkpoint"):
        asyncio.run(execution.validate(source, output))

    assert native_calls == ["one", "two"]
    assert len(list((execution._state_dir(output) / "reviews").glob("*.json"))) == 2
    saved = Document.model_validate_json(output.read_text())
    assert "grow_roots.discover_root_locators.hunt_docket_locators" not in saved.substage_runs

    final = asyncio.run(execution.resume(output))

    assert native_calls == ["one", "two", "three"]
    assert resumed_results == first_results
    assert first_results[1]["success"] is False
    assert reused == [2]
    assert len(list((execution._state_dir(output) / "reviews").glob("*.json"))) == 3
    assert "validate_all_citation_content.review_citation_content" in final.stage_runs


def _body_provider_result(monkeypatch, source, calls, failure):
    """Install one real reporter root and a typed provider result in real composers."""
    from mellea_lrc.extraction.find_full_reporter_locators import find_full_reporter_locators
    from mellea_lrc.model.citations.body_evidence import BodySearch, BodySource

    source.write_text("See Brown v. Board of Education, 347 U.S. 483 (1954).", encoding="utf-8")
    _fake_atomic_operations(monkeypatch, calls)
    reporter_substage = "grow_roots.discover_root_locators.find_full_reporter_locators"
    form_substage = "grow_roots.form_root_groups.form_roots"
    body_substage = (
        "validate_roots_citation_identity.corroborate_locator_bodies.retrieve_courtlistener_opinions"
    )
    attempts = 0

    def reporters(document):
        calls.append(reporter_substage)
        return find_full_reporter_locators(document)

    def roots(document):
        calls.append(form_substage)
        for citation in document.full_locators:
            document = document.replace_citation(citation.record(form_substage).with_root(citation.id))
        return document.complete_substage(form_substage)

    def body(document, **kwargs):
        nonlocal attempts
        attempts += 1
        calls.append(body_substage)
        root = document.roots[0].record(body_substage)
        result = BodySearch(
            node_id=root.nodes[-1].id,
            source=BodySource.COURTLISTENER_OPINION,
            retrospective_date=None,
            failures=(failure,) if attempts == 1 else (),
        )
        return document.replace_citation(root.with_body_search(result)).complete_substage(body_substage)

    discover_stage = importlib.import_module("mellea_lrc.workflows.grow_roots.discover_root_locators")
    form_stage = importlib.import_module("mellea_lrc.workflows.grow_roots.form_root_groups")
    body_stage = importlib.import_module(
        "mellea_lrc.workflows.validate_roots_citation_identity.corroborate_locator_bodies"
    )
    monkeypatch.setattr(discover_stage, "find_full_reporter_locators", reporters)
    monkeypatch.setattr(form_stage, "form_roots", roots)
    monkeypatch.setattr(body_stage, "retrieve_courtlistener_locator_opinions", body)
    return body_substage


@pytest.mark.parametrize(
    ("failure_type", "status"), [("transport_error", None), ("http_error", 429), ("http_error", 503)]
)
def test_transient_provider_result_preserves_last_good_checkpoint_and_retries_only_failed_step(
    execution, source_and_output, monkeypatch, failure_type, status
):
    from mellea_lrc.model.citations.body_evidence import BodyEvidenceFailure

    source, output = source_and_output
    failure = BodyEvidenceFailure(
        failure_type=failure_type,
        message="Provider temporarily unavailable",
        status_code=status,
    )
    calls = []
    substage = _body_provider_result(monkeypatch, source, calls, failure)

    with pytest.raises(RuntimeError, match=r"(?i)(provider|transient|retrieval)"):
        asyncio.run(execution.validate(source, output))

    saved = Document.model_validate_json(output.read_text())
    assert substage not in saved.substage_runs
    assert saved.roots[0].body_searches == ()
    failures = list((execution._state_dir(output) / "failed-checkpoints").glob("*.json"))
    assert len(failures) == 1
    failed = Document.model_validate_json(failures[0].read_text())
    assert failed.substage_runs[-1] == substage
    assert failed.roots[0].body_searches[-1].failures == (failure,)
    committed = set(saved.substage_runs)
    calls.clear()

    final = asyncio.run(execution.resume(output))

    assert calls[0] == substage
    assert not committed.intersection(calls)
    assert calls.count(substage) == 1
    assert final.roots[0].body_searches[-1].failures == ()
    assert "validate_all_citation_content.review_citation_content" in final.stage_runs
    assert Document.model_validate_json(failures[0].read_text()) == failed


def test_genuine_missing_body_does_not_interrupt_or_trigger_retry(execution, source_and_output, monkeypatch):
    from mellea_lrc.model.citations.body_evidence import BodyEvidenceFailure

    source, output = source_and_output
    failure = BodyEvidenceFailure(failure_type="http_error", message="Body does not exist", status_code=404)
    calls = []
    substage = _body_provider_result(monkeypatch, source, calls, failure)

    final = asyncio.run(execution.validate(source, output))

    assert final.roots[0].body_searches[-1].failures == (failure,)
    assert calls.count(substage) == 1
    assert not (execution._state_dir(output) / "failed-checkpoints").exists()
    assert "validate_all_citation_content.review_citation_content" in final.stage_runs


def test_configuration_drift_before_checkpoint_does_not_replace_last_good_document(
    execution, source_and_output, monkeypatch
):
    source, output = source_and_output
    settings = {"field_policy": "RULE"}
    monkeypatch.setattr(execution, "_configuration", lambda fields: dict(settings))
    calls = []
    _fake_atomic_operations(monkeypatch, calls)
    substage = "grow_roots.discover_root_locators.hunt_docket_locators"
    attempts = 0

    async def change_configuration(document, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            settings["field_policy"] = "ML"
        return document.complete_substage(substage)

    stage = importlib.import_module("mellea_lrc.workflows.grow_roots.discover_root_locators")
    monkeypatch.setattr(stage, "hunt_docket_locators", change_configuration)
    with pytest.raises(ValueError, match=r"(?i)configuration changed"):
        asyncio.run(execution.validate(source, output))

    saved = Document.model_validate_json(output.read_text())
    assert saved.substage_runs == (
        "grow_roots.discover_root_locators.find_full_reporter_locators",
        "grow_roots.discover_root_locators.find_docket_locators",
    )
    assert json.loads((execution._state_dir(output) / "run.json").read_text())["status"] == "interrupted"
    settings["field_policy"] = "RULE"
    calls.clear()

    final = asyncio.run(execution.resume(output))

    assert attempts == 2
    assert not set(saved.substage_runs).intersection(calls)
    assert "validate_all_citation_content.review_citation_content" in final.stage_runs


def test_configuration_drift_between_reviews_stops_before_next_provider_call(
    execution, source_and_output, monkeypatch
):
    from mellea.stdlib.sampling import MultiTurnStrategy

    from mellea_lrc.llm import ivr
    from mellea_lrc.llm.journal import active_journal
    from mellea_lrc.model.ivr import IvrAttempt, IvrRun

    source, output = source_and_output
    settings = {"field_policy": "RULE"}
    monkeypatch.setattr(execution, "_configuration", lambda fields: dict(settings))
    _fake_atomic_operations(monkeypatch, [])
    native_calls = []
    session = SimpleNamespace(backend=SimpleNamespace(model_id="test-model"))
    strategy = MultiTurnStrategy(loop_budget=3)
    reused = []

    async def native(session, spec, *, strategy, model_options):
        native_calls.append(spec.user_variables["citation"])
        result = IvrRun(
            success=True,
            selected_attempt=0,
            attempts=(IvrAttempt(output="{}", requirements=()),),
            **ivr._run_metadata(session, spec, model_options),
        )
        if len(native_calls) == 1:
            settings["field_policy"] = "ML"
        return result

    async def review_sites(document, **kwargs):
        for identifier in ("one", "two"):
            await ivr.run_instruct_ivr(
                session,
                ivr.InstructIvrSpec(
                    description="Review this citation against the source.",
                    grounding_context={"source": document.text},
                    user_variables={"citation": identifier},
                ),
                strategy=strategy,
                model_options={},
            )
        reused.append(active_journal().reused)
        return document.complete_substage("grow_roots.discover_root_locators.hunt_docket_locators")

    monkeypatch.setattr(ivr, "_run_instruct_ivr", native)
    stage = importlib.import_module("mellea_lrc.workflows.grow_roots.discover_root_locators")
    monkeypatch.setattr(stage, "hunt_docket_locators", review_sites)
    with pytest.raises(ValueError, match=r"(?i)configuration changed"):
        asyncio.run(execution.validate(source, output))

    assert native_calls == ["one"]
    assert len(list((execution._state_dir(output) / "reviews").glob("*.json"))) == 1
    saved = Document.model_validate_json(output.read_text())
    assert "grow_roots.discover_root_locators.hunt_docket_locators" not in saved.substage_runs
    assert json.loads((execution._state_dir(output) / "run.json").read_text())["status"] == "interrupted"
    settings["field_policy"] = "RULE"

    final = asyncio.run(execution.resume(output))

    assert native_calls == ["one", "two"]
    assert reused == [1]
    assert "validate_all_citation_content.review_citation_content" in final.stage_runs
