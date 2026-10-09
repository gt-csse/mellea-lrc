"""The four workflows, each composed from independently callable stages."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from mellea_lrc.model.document import Document
from mellea_lrc.model.execution import get_stage_definition, get_workflow

Checkpoint = Callable[[Document], None]


def _save_checkpoint(document: Document, checkpoint: Checkpoint | None) -> None:
    if checkpoint is not None:
        checkpoint(document)


def _require_prefix(document: Document, substages: tuple[str, ...], name: str) -> None:
    completed = tuple(substage for substage in document.substage_runs if substage in substages)
    if completed != substages[: len(completed)]:
        raise ValueError(f"{name} checkpoint must end at a completed substage boundary")


def _require_stage_prefix(document: Document, stage: str, *, omitted_substages: tuple[str, ...] = ()) -> None:
    """Reject duplicate groups and holes before running any atomic operation."""
    if stage in document.stage_runs:
        raise ValueError(f"Stage already completed: {stage}")
    substages = tuple(
        item.name
        for item in get_stage_definition(stage).substages
        if item.name not in omitted_substages or item.name in document.substage_runs
    )
    _require_prefix(document, substages, stage)


def _require_workflow_prefix(
    document: Document,
    workflow: str,
    *,
    omitted_substages: tuple[str, ...] = (),
) -> None:
    """Validate a workflow's atomic and group histories before resuming."""
    definition = get_workflow(workflow)
    substages = tuple(
        item.name
        for stage in definition.stages
        for item in stage.substages
        if item.name not in omitted_substages or item.name in document.substage_runs
    )
    stages = tuple(stage.name for stage in definition.stages)
    completed = tuple(stage for stage in document.stage_runs if stage in stages)
    if completed != stages[: len(completed)]:
        raise ValueError(f"{workflow} checkpoint must end at a completed stage boundary")
    for index, stage in enumerate(definition.stages):
        if stage.name not in completed:
            later = {
                item.name for following in definition.stages[index + 1 :] for item in following.substages
            }
            if later.intersection(document.substage_runs):
                raise ValueError(
                    f"{workflow} checkpoint is missing stage {stage.name} before later substages"
                )
        if stage.name in completed:
            missing = tuple(
                item.name
                for item in stage.substages
                if item.name not in omitted_substages and item.name not in document.substage_runs
            )
            if missing:
                raise ValueError(f"Completed stage {stage.name} is missing requested substages: {missing}")
    _require_prefix(document, substages, workflow)


def _require_body_search_cutoff(document: Document, cutoff: date | None) -> None:
    """Reject resumed evidence collected under a different eligibility cutoff."""
    for citation in document.full_locators:
        for search in (*citation.body_searches, *citation.field_body_searches):
            if search.retrospective_date != cutoff:
                raise ValueError(
                    f"Saved body search cutoff differs for {document.source_path}: "
                    f"{search.retrospective_date} != {cutoff}"
                )


def _finish_stage(document: Document, stage: str, checkpoint: Checkpoint | None) -> Document:
    document = document.complete_stage(stage)
    _save_checkpoint(document, checkpoint)
    return document


from mellea_lrc.workflows.grow_leaves import grow_leaves  # noqa: E402
from mellea_lrc.workflows.grow_roots import grow_roots  # noqa: E402
from mellea_lrc.workflows.validate_pincite import validate_pincite  # noqa: E402
from mellea_lrc.workflows.validate_roots import validate_roots  # noqa: E402

__all__ = ["grow_leaves", "grow_roots", "validate_pincite", "validate_roots"]
