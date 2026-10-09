"""Semantic groups preserve exact boundaries and resume atomic operations."""

from __future__ import annotations

import asyncio
import importlib
import inspect

import pytest

from mellea_lrc import api
from mellea_lrc.model.document import Document
from mellea_lrc.model.execution import WORKFLOW_CATALOG, get_workflow

STAGE_FUNCTIONS = {
    "grow_roots.locator_discovery": "discover_root_locators",
    "grow_roots.field_reading": "read_root_fields",
    "grow_roots.root_formation": "form_root_groups",
    "validate_roots.reporter_lookup": "lookup_reporter_roots",
    "validate_roots.docket_lookup": "lookup_docket_roots",
    "validate_roots.locator_body_corroboration": "corroborate_locator_bodies",
    "validate_roots.intended_case_discovery": "discover_intended_cases",
    "grow_leaves.short_reporter_citations": "grow_short_reporter_leaves",
    "grow_leaves.reference_citations": "grow_reference_leaves",
    "grow_leaves.id_citations": "grow_id_leaves",
    "grow_leaves.supra_citations": "grow_supra_leaves",
    "grow_leaves.leaf_field_correction": "correct_leaf_readings",
    "validate_pincite.opinion_preparation": "prepare_root_opinions",
    "validate_pincite.citation_preparation": "prepare_citation_evidence",
    "validate_pincite.support_review": "review_citation_support",
}


def _invoke(function, document, **kwargs):
    result = function(document, **kwargs)
    return asyncio.run(result) if inspect.isawaitable(result) else result


def _fake_substages(monkeypatch, calls):
    def asynchronous(substage):
        async def run(document, **kwargs):
            calls.append(substage)
            return document.complete_substage(substage)

        return run

    def with_positional_options(substage):
        def run(document, *args, **kwargs):
            calls.append(substage)
            return document.complete_substage(substage)

        return run

    for workflow in WORKFLOW_CATALOG:
        for stage in workflow.stages:
            module = importlib.import_module(f"mellea_lrc.workflows.{stage.name}")
            replaced = set()
            for name, function in vars(module).copy().items():
                if not inspect.isfunction(function) or not function.__module__.startswith(
                    ("mellea_lrc.extraction.", "mellea_lrc.validation.")
                ):
                    continue
                atomic = importlib.import_module(function.__module__)
                substage = getattr(atomic, "SUBSTAGE", None)
                if substage is None:
                    continue
                factory = asynchronous if inspect.iscoroutinefunction(function) else with_positional_options
                monkeypatch.setattr(module, name, factory(substage))
                replaced.add(substage)
            assert replaced == {item.name for item in stage.substages}


@pytest.mark.parametrize("workflow_name", [item.name for item in WORKFLOW_CATALOG])
def test_every_semantic_checkpoint_recovers_exact_json_after_later_substages(monkeypatch, workflow_name):
    calls = []
    _fake_substages(monkeypatch, calls)
    saved = []
    options = {
        "grow_roots": {"hunt_dockets": True, "review_docket_roots": True},
        "validate_roots": {"search_other_fields": True},
    }.get(workflow_name, {})
    final = _invoke(
        getattr(api, workflow_name), Document.from_source("No citations."), checkpoint=saved.append, **options
    )
    expected = get_workflow(workflow_name)
    assert final.stage_runs == tuple(stage.name for stage in expected.stages)
    grouped = {item.runs[-1].name: item.model_dump_json() for item in saved if item.runs[-1].kind == "stage"}
    assert set(grouped) == set(final.stage_runs)
    reloaded = Document.model_validate_json(final.model_dump_json())
    for stage in expected.stages:
        checkpoint = reloaded.get_stage(stage.name)
        assert checkpoint.model_dump_json() == grouped[stage.name]
        assert checkpoint.runs[-1].kind == "stage"
        assert checkpoint.runs[-1].name == stage.name
    assert final.citations == ()


@pytest.mark.parametrize("workflow_name", [item.name for item in WORKFLOW_CATALOG])
def test_every_partial_atomic_checkpoint_resumes_without_replaying_prior_operations(
    monkeypatch, workflow_name
):
    calls = []
    _fake_substages(monkeypatch, calls)
    saved = []
    options = {
        "grow_roots": {"hunt_dockets": True, "review_docket_roots": True},
        "validate_roots": {"search_other_fields": True},
    }.get(workflow_name, {})
    run = getattr(api, workflow_name)
    final = _invoke(run, Document.from_source("No citations."), checkpoint=saved.append, **options)
    ordered_calls = tuple(calls)
    for checkpoint in saved:
        checkpoint = Document.model_validate_json(checkpoint.model_dump_json())
        completed = len(checkpoint.substage_runs)
        calls.clear()
        resumed = _invoke(run, checkpoint, **options)
        assert tuple(calls) == ordered_calls[completed:]
        assert resumed.model_dump_json() == final.model_dump_json()
    calls.clear()
    assert _invoke(run, final, **options) is final
    assert calls == []


@pytest.mark.parametrize("stage_name", STAGE_FUNCTIONS)
def test_completed_semantic_stages_reject_direct_reexecution(monkeypatch, stage_name):
    calls = []
    _fake_substages(monkeypatch, calls)
    run = getattr(api, STAGE_FUNCTIONS[stage_name])
    completed = _invoke(run, Document.from_source("No citations."))
    assert completed.stage_runs == (stage_name,)
    calls.clear()
    with pytest.raises(ValueError, match="Stage already completed"):
        _invoke(run, completed)
    assert calls == []


def test_completed_atomic_group_without_marker_commits_group_without_retrieval(monkeypatch):
    calls = []
    _fake_substages(monkeypatch, calls)
    stage = get_workflow("validate_roots").stages[0]
    document = Document.from_source("No citations.")
    for substage in stage.substages:
        document = document.complete_substage(substage.name)
    checkpoints = []
    result = _invoke(api.lookup_reporter_roots, document, checkpoint=checkpoints.append)
    assert calls == []
    assert result.stage_runs == (stage.name,)
    assert checkpoints == [result]
    assert result.get_stage(stage.name) == result
    assert result.get_substage(stage.substages[-1].name) == document


def test_optional_group_and_substages_are_omitted_without_completion_markers(monkeypatch):
    calls = []
    _fake_substages(monkeypatch, calls)
    document = _invoke(api.grow_roots, Document.from_source("No citations."))
    document = _invoke(api.validate_roots, document)
    document = _invoke(api.grow_leaves, document, review_leaves=False)
    assert "grow_roots.locator_discovery.docket_hunting" not in document.substage_runs
    assert "grow_roots.root_formation.docket_llm_reassignment" not in document.substage_runs
    assert "grow_leaves.supra_citations.llm_attribution" not in document.substage_runs
    assert "validate_roots.intended_case_discovery" not in document.stage_runs
    assert not any(name.startswith("validate_roots.intended_case_discovery.") for name in calls)
    with pytest.raises(KeyError, match="has not run"):
        document.get_stage("validate_roots.intended_case_discovery")


def test_workflow_rejects_holes_and_missing_prior_group_markers_before_calls(monkeypatch):
    calls = []
    _fake_substages(monkeypatch, calls)
    reporter, docket, *_ = get_workflow("validate_roots").stages
    document = Document.from_source("No citations.").complete_substage(reporter.substages[1].name)
    with pytest.raises(ValueError, match="completed substage boundary"):
        _invoke(api.validate_roots, document)
    document = Document.from_source("No citations.")
    for substage in reporter.substages:
        document = document.complete_substage(substage.name)
    document = document.complete_substage(docket.substages[0].name)
    with pytest.raises(ValueError, match="missing stage"):
        _invoke(api.validate_roots, document)
    assert calls == []


def test_enabled_optional_operation_cannot_be_added_to_committed_stage(monkeypatch):
    calls = []
    _fake_substages(monkeypatch, calls)
    document = _invoke(api.grow_roots, Document.from_source("No citations."))
    calls.clear()
    with pytest.raises(ValueError, match="missing requested substages"):
        _invoke(api.grow_roots, document, hunt_dockets=True)
    assert calls == []


def test_real_zero_citation_workflows_commit_all_requested_groups():
    async def run():
        document = Document.from_source("No citations here.")
        saved = []
        document = await api.grow_roots(document, checkpoint=saved.append)
        document = await api.validate_roots(document, search_other_fields=True, checkpoint=saved.append)
        document = await api.grow_leaves(document, checkpoint=saved.append)
        document = await api.validate_pincite(document, checkpoint=saved.append)
        return document, saved

    final, saved = asyncio.run(run())
    assert len(final.stage_runs) == 15
    assert final.citations == ()
    for checkpoint in saved:
        run = checkpoint.runs[-1]
        recovered = final.get_stage(run.name) if run.kind == "stage" else final.get_substage(run.name)
        assert recovered.model_dump_json() == checkpoint.model_dump_json()


def test_real_group_snapshots_survive_later_citation_fields_and_attachments():
    async def run():
        saved = []
        document = Document.from_source("Alpha v. Beta, 123 F.3d 456, 459 (2d Cir. 2001). Id. at 460.")
        document = await api.grow_roots(document, checkpoint=saved.append)
        document = await api.grow_leaves(document, review_leaves=False, checkpoint=saved.append)
        return document, saved

    final, saved = asyncio.run(run())
    assert len(final.citations) == 2
    assert final.leaves[0].root_id[-1].value == final.roots[0].id
    restored = Document.model_validate_json(final.model_dump_json())
    groups = [item for item in saved if item.runs[-1].kind == "stage"]
    assert len(groups) == 8
    for checkpoint in groups:
        assert restored.get_stage(checkpoint.runs[-1].name).model_dump_json() == checkpoint.model_dump_json()
    discovery = restored.get_stage("grow_roots.locator_discovery")
    assert len(discovery.citations) == 1
    assert discovery.citations[0].case_name == ()
    assert discovery.leaves == ()


@pytest.mark.parametrize("source", ["courtlistener_opinion", "courtlistener_recap", "govinfo_opinion"])
@pytest.mark.parametrize(
    "entrypoint,field_search,completed_group",
    [
        ("validate_roots", False, False),
        ("validate_roots", False, True),
        ("validate_roots", True, False),
        ("validate_roots", True, True),
        ("corroborate_locator_bodies", False, False),
        ("discover_intended_cases", True, False),
    ],
)
def test_body_resume_rejects_changed_cutoff_before_saved_evidence_reaches_review(
    monkeypatch, source, entrypoint, field_search, completed_group
):
    from datetime import date

    from mellea_lrc.model.citations.body_evidence import BodyEvidence, BodySearch, BodySource
    from mellea_lrc.model.citations.field_body_evidence import FieldBodySearch
    from mellea_lrc.model.span import Span

    document = _invoke(api.grow_roots, Document.from_source("Alpha v. Beta, 123 F.3d 456 (2001)."))
    calls = []
    _fake_substages(monkeypatch, calls)
    groups = get_workflow("validate_roots").stages
    target = groups[3 if field_search else 2]
    for stage in groups[: groups.index(target)]:
        document = _invoke(getattr(api, STAGE_FUNCTIONS[stage.name]), document)
    substage = target.substages[0].name
    root = document.roots[0].record(substage)
    excerpt = "Alpha v. Beta" if field_search else "123 F.3d 456"
    evidence = BodyEvidence(
        body_id="saved-body",
        issued_on=date(2015, 1, 1),
        body_sha256="a" * 64,
        excerpt=excerpt,
        source_offset=0,
        anchor_kind="case_name" if field_search else "locator",
        anchor_span=Span(start=0, end=len(excerpt)),
    )
    search_kwargs = {
        "node_id": root.nodes[-1].id,
        "source": BodySource(source),
        "retrospective_date": date(2020, 1, 1),
        "evidence": (evidence,),
    }
    if field_search:
        root = root.with_field_body_search(FieldBodySearch(query_name=excerpt, **search_kwargs))
    else:
        root = root.with_body_search(BodySearch(**search_kwargs))
    document = document.replace_citation(root).complete_substage(substage)
    if completed_group:
        for item in target.substages[1:]:
            document = document.complete_substage(item.name)
        document = document.complete_stage(target.name)
    calls.clear()
    options = {"search_other_fields": field_search} if entrypoint == "validate_roots" else {}
    with pytest.raises(ValueError, match="Saved body search cutoff differs"):
        _invoke(getattr(api, entrypoint), document, retrospective_date=date(2010, 1, 1), **options)
    assert calls == []
