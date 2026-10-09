"""Semantic grouping preserves exact, native atomic and group checkpoints."""

import json
from dataclasses import FrozenInstanceError

import pytest

from mellea_lrc.model import (
    STAGE_CATALOG,
    SUBSTAGE_CATALOG,
    WORKFLOW_CATALOG,
    CheckpointRun,
    Document,
    FullDocketCitation,
    Span,
    TableOfAuthoritiesComponent,
    get_stage_definition,
    get_substage_definition,
    get_workflow,
    stage_for_substage,
)
from mellea_lrc.model.site_review import SiteReview

DISCOVERY = "grow_roots.locator_discovery"
REPORTERS = f"{DISCOVERY}.full_reporter_locators"
DOCKETS = f"{DISCOVERY}.docket_locators"
HUNTING = f"{DISCOVERY}.docket_hunting"
ROOTS = "grow_roots.root_formation"
RULE = f"{ROOTS}.rule"
REASSIGNMENT = f"{ROOTS}.docket_llm_reassignment"
TEXT = "No. 1:24-cv-00123; No. 2:24-cv-00456."


def _docket(substage: str, number: str = "1:24-cv-00123") -> FullDocketCitation:
    start = TEXT.index(f"No. {number}")
    return FullDocketCitation.from_locator(
        citation_id=f"docket:{number}",
        substage=substage,
        source=TEXT,
        span=Span(start, start + 4 + len(number)),
        number_span=Span(start + 4, start + 4 + len(number)),
    )


def _discovered(*, hunting: bool = False) -> Document:
    document = Document.from_source(TEXT).complete_substage(REPORTERS)
    document = document.add_citation(_docket(DOCKETS)).complete_substage(DOCKETS)
    if hunting:
        document = document.add_site_review(
            SiteReview(
                substage=HUNTING,
                candidate_span=Span(0, 3),
                candidate_text="No.",
                outcome="declined",
                reason="Existing grounded docket citation covers the site.",
            )
        ).complete_substage(HUNTING)
    return document


def test_catalog_is_immutable_and_covers_four_workflows_fifteen_stages() -> None:
    assert tuple(workflow.name for workflow in WORKFLOW_CATALOG) == (
        "grow_roots",
        "validate_roots",
        "grow_leaves",
        "validate_pincite",
    )
    assert len(STAGE_CATALOG) == 15
    assert len(SUBSTAGE_CATALOG) == 53
    stage = get_stage_definition(DISCOVERY)
    assert stage.required_substages == (REPORTERS, DOCKETS)
    assert tuple(substage.name for substage in stage.substages) == (REPORTERS, DOCKETS, HUNTING)
    assert get_substage_definition(HUNTING).optional
    assert stage_for_substage(HUNTING) == DISCOVERY
    assert stage_for_substage("development_checkpoint") is None
    assert get_workflow("validate_roots").required_stages == (
        "validate_roots.reporter_lookup",
        "validate_roots.docket_lookup",
        "validate_roots.locator_body_corroboration",
    )
    with pytest.raises(TypeError):
        STAGE_CATALOG["new"] = stage
    with pytest.raises(FrozenInstanceError):
        stage.optional = True
    with pytest.raises(KeyError, match="Unknown workflow"):
        get_workflow("missing")
    with pytest.raises(KeyError, match="Unknown substage"):
        get_substage_definition("missing")


@pytest.mark.parametrize("hunting", [False, True])
def test_group_and_atomic_recovery_keep_distinct_exact_event_boundaries(hunting: bool) -> None:
    atomic = _discovered(hunting=hunting)
    completed = atomic.complete_stage(DISCOVERY)
    assert completed.citations == atomic.citations
    assert completed.site_reviews == atomic.site_reviews
    assert completed.runs == (*atomic.runs, CheckpointRun(kind="stage", name=DISCOVERY))
    assert completed.stage_runs == (DISCOVERY,)
    assert completed.substage_runs == atomic.substage_runs
    assert completed != atomic

    original = completed.citations[0]
    rooted = completed.replace_citation(original.record(RULE).with_root(original.id)).complete_substage(RULE)
    final = rooted.complete_stage(ROOTS).complete_substage("after_groups_noop")
    loaded = Document.model_validate_json(final.model_dump_json())
    for document in (final, loaded):
        assert document.get_stage(DISCOVERY) == completed
        assert document.get_substage(HUNTING if hunting else DOCKETS) == atomic
        assert document.get_substage(RULE) == rooted
        assert document.get_stage(ROOTS) == rooted.complete_stage(ROOTS)
        assert document.get_substage("after_groups_noop") == final
        assert document.get_substage(REPORTERS).citations == ()
    assert loaded == final


def test_native_serialization_persists_only_one_typed_run_log() -> None:
    document = _discovered().complete_stage(DISCOVERY)
    data = json.loads(document.model_dump_json())
    assert "stage_runs" not in data
    assert "substage_runs" not in data
    assert data["runs"] == [
        {"kind": "substage", "name": REPORTERS},
        {"kind": "substage", "name": DOCKETS},
        {"kind": "stage", "name": DISCOVERY},
    ]
    for legacy_name in ("stage_runs", "substage_runs"):
        with pytest.raises(ValueError, match="Extra inputs"):
            Document.model_validate({**data, legacy_name: []})
    with pytest.raises(ValueError, match="frozen"):
        document.runs[0].name = "changed"


def test_partial_stage_has_no_recoverable_group_boundary() -> None:
    partial = Document.from_source(TEXT).complete_substage(REPORTERS)
    with pytest.raises(KeyError, match="has not run"):
        partial.get_stage(DISCOVERY)
    with pytest.raises(ValueError, match="unfinished required substages"):
        partial.complete_stage(DISCOVERY)
    with pytest.raises(KeyError, match="Unknown stage"):
        partial.complete_stage("unknown_group")
    with pytest.raises(KeyError, match="Unknown stage"):
        partial.get_stage("unknown_group")
    with pytest.raises(KeyError, match="has not run"):
        partial.get_substage("unknown_atomic")


def test_completed_group_cannot_gain_optional_substages_or_citation_work() -> None:
    completed = _discovered().complete_stage(DISCOVERY)
    with pytest.raises(ValueError, match="already completed"):
        completed.complete_stage(DISCOVERY)
    with pytest.raises(ValueError, match="completed stage"):
        completed.complete_substage(HUNTING)
    with pytest.raises(ValueError, match="completed stage"):
        completed.add_citation(_docket(HUNTING, "2:24-cv-00456"))
    with pytest.raises(ValueError, match="completed stage"):
        completed.replace_citation(completed.citations[0].record(HUNTING))
    with pytest.raises(ValueError, match="completed stage"):
        completed.add_site_review(
            SiteReview(
                substage=HUNTING,
                candidate_span=Span(0, 3),
                candidate_text="No.",
                outcome="failed",
                reason="No review result.",
            )
        )


def test_required_substages_of_optional_workflow_stage_still_required() -> None:
    intended = get_stage_definition("validate_roots.intended_case_discovery")
    assert intended.optional
    assert len(intended.required_substages) == 4
    partial = Document.from_source(TEXT).complete_substage(intended.required_substages[0])
    with pytest.raises(ValueError, match="unfinished required substages"):
        partial.complete_stage(intended.name)


def test_completion_rejects_pending_work_and_preserves_unacted_citations() -> None:
    document = _discovered()
    original = document.citations[0]
    pending = document.replace_citation(original.record("manual_review"))
    with pytest.raises(ValueError, match="pending substage"):
        pending.complete_stage(DISCOVERY)
    with pytest.raises(ValueError, match="pending substage"):
        pending.complete_substage("other_review")
    assert pending.get_substage(DOCKETS) == document
    completed = pending.complete_substage("manual_review")
    with pytest.raises(ValueError, match="immediately after its own final substage"):
        completed.complete_stage(DISCOVERY)
    assert completed.get_substage(DOCKETS).citations[0] == original
    assert len(completed.citations[0].nodes) == 2


def test_group_boundaries_preserve_component_tags_and_relationship_history() -> None:
    empty = Document.from_source(TEXT)
    empty = Document.model_validate(
        {**empty.model_dump(mode="python"), "index_spans": (TableOfAuthoritiesComponent(0, len(TEXT)),)}
    )
    discovered = empty.complete_substage(REPORTERS).add_citation(_docket(DOCKETS)).complete_substage(DOCKETS)
    group = discovered.complete_stage(DISCOVERY)
    assert len(group.citations[0].tags) == 1
    root = group.citations[0]
    rooted = group.replace_citation(root.record(RULE).with_root(root.id)).complete_substage(RULE)
    root_group = rooted.complete_stage(ROOTS)
    withdrawn = root_group.replace_citation(rooted.citations[0].record("manual_withdrawal").withdraw())
    loaded = Document.model_validate_json(withdrawn.model_dump_json())
    assert loaded.get_stage(DISCOVERY) == group
    assert loaded.get_stage(ROOTS) == root_group
    assert loaded.citations[0].tags == group.citations[0].tags


@pytest.mark.parametrize(
    ("runs", "message"),
    [
        ([CheckpointRun(kind="stage", name=DISCOVERY)], "unfinished"),
        (
            [
                CheckpointRun(kind="substage", name=DOCKETS),
                CheckpointRun(kind="substage", name=REPORTERS),
                CheckpointRun(kind="stage", name=DISCOVERY),
            ],
            "catalog order",
        ),
        (
            [
                CheckpointRun(kind="substage", name=REPORTERS),
                CheckpointRun(kind="substage", name=DOCKETS),
                CheckpointRun(kind="stage", name=DISCOVERY),
                CheckpointRun(kind="substage", name=HUNTING),
            ],
            "after its completed stage",
        ),
        (
            [
                CheckpointRun(kind="substage", name=RULE),
                CheckpointRun(kind="stage", name=ROOTS),
                CheckpointRun(kind="stage", name=ROOTS),
            ],
            "Stage runs must be unique",
        ),
        ([CheckpointRun(kind="substage", name=RULE), CheckpointRun(kind="substage", name=RULE)], "unique"),
        ([CheckpointRun(kind="stage", name="unknown_group")], "Unknown stage"),
        (
            [
                CheckpointRun(kind="substage", name=REPORTERS),
                CheckpointRun(kind="substage", name=DOCKETS),
                CheckpointRun(kind="substage", name=RULE),
                CheckpointRun(kind="stage", name=DISCOVERY),
            ],
            "immediately after its own final substage",
        ),
    ],
)
def test_native_reload_rejects_corrupt_run_boundaries(runs: list[CheckpointRun], message: str) -> None:
    data = Document.from_source(TEXT).model_dump(mode="python")
    with pytest.raises(ValueError, match=message):
        Document.model_validate({**data, "runs": runs})


def test_native_reload_rejects_pending_work_inside_completed_group() -> None:
    document = _discovered().complete_stage(DISCOVERY)
    data = document.model_dump(mode="python")
    data["citations"][0]["nodes"] = (
        *data["citations"][0]["nodes"],
        {"id": "optional-late-node", "substage": HUNTING},
    )
    with pytest.raises(ValueError, match="pending substage changes"):
        Document.model_validate(data)


def test_native_reload_rejects_citation_nodes_moving_backward() -> None:
    document = _discovered().complete_substage("manual_review")
    data = document.model_dump(mode="python")
    data["citations"][0]["nodes"] = (
        *data["citations"][0]["nodes"],
        {"id": "manual-node", "substage": "manual_review"},
        {"id": "backward-node", "substage": DOCKETS},
    )
    with pytest.raises(ValueError, match="move backward"):
        Document.model_validate(data)


def test_catalog_order_is_checked_when_optional_substage_is_enabled() -> None:
    document = Document.from_source(TEXT).complete_substage(REPORTERS).complete_substage(HUNTING)
    document = document.complete_substage(DOCKETS)
    with pytest.raises(ValueError, match="catalog order"):
        document.complete_stage(DISCOVERY)


def test_noop_optional_run_is_distinct_and_included_when_enabled() -> None:
    document = Document.from_source(TEXT).complete_substage(RULE)
    enabled = document.complete_substage(REASSIGNMENT)
    completed = enabled.complete_stage(ROOTS)
    assert completed.citations == ()
    assert completed.get_substage(RULE) == document
    assert completed.get_substage(REASSIGNMENT) == enabled
    assert completed.get_stage(ROOTS) == completed


def test_group_of_noop_substages_does_not_add_nodes_to_existing_citations() -> None:
    discovered = _discovered().complete_stage(DISCOVERY)
    stage = get_stage_definition("grow_roots.field_reading")
    document = discovered
    for substage in stage.substages:
        document = document.complete_substage(substage.name)
    completed = document.complete_stage(stage.name)
    assert completed.citations == discovered.citations
    assert len(completed.citations[0].nodes) == 1
    assert completed.get_stage(DISCOVERY) == discovered
    assert completed.get_stage(stage.name) == completed


def test_native_reload_rejects_future_root_relationship_across_group_boundary() -> None:
    discovered = _discovered().complete_stage(DISCOVERY)
    document = discovered.add_citation(_docket("later_discovery", "2:24-cv-00456"))
    document = document.complete_substage("later_discovery")
    data = document.model_dump(mode="python")
    data["citations"][0]["root_id"] = [
        {"node_id": document.citations[0].nodes[0].id, "value": document.citations[1].id}
    ]
    with pytest.raises(ValueError, match="created after its assignment"):
        Document.model_validate(data)


def test_complete_stage_cannot_fabricate_boundary_after_later_group_work() -> None:
    document = _discovered().complete_substage(RULE)
    with pytest.raises(ValueError, match="immediately after its own final substage"):
        document.complete_stage(DISCOVERY)
    assert document.get_substage(RULE) == document
    with pytest.raises(KeyError, match="has not run"):
        document.get_stage(DISCOVERY)
