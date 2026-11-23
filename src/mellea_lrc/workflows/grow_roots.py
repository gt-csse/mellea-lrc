"""Discover, read, and group full citation occurrences."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.extraction import (
    find_docket_locators,
    find_full_reporter_locators,
    form_roots,
    resolve_case_names,
    resolve_colocations,
    resolve_courts,
    resolve_dates,
    resolve_docket_entries,
    resolve_pin_cites,
)
from mellea_lrc.model.document import Document
from mellea_lrc.model.execution import GROW_ROOTS_RUNS, GROW_ROOTS_STAGES, StageDefinition, grow_roots_runs

Checkpoint = Callable[[Document], None]
_DISCOVERY, _READING, _GROUPING = GROW_ROOTS_STAGES


def _require_resume_prefix(document: Document) -> None:
    completed = grow_roots_runs(document.runs)
    if completed != GROW_ROOTS_RUNS[: len(completed)]:
        raise ValueError("grow_roots requires an ordered workflow checkpoint")
    if document._pending_substages():
        raise ValueError("Resume grow_roots from a completed substage checkpoint")


def _run_stage(
    document: Document,
    stage: StageDefinition,
    writers: tuple[Callable[[Document], Document], ...],
    checkpoint: Checkpoint | None,
) -> Document:
    _require_resume_prefix(document)
    if stage.name in document.stage_runs:
        raise ValueError(f"Stage already completed: {stage.name}")
    if len(document.stage_runs) != GROW_ROOTS_STAGES.index(stage):
        raise ValueError(f"Complete the preceding stages before {stage.name}")

    completed = sum(name in document.substage_runs for name in stage.substages)
    for write in writers[completed:]:
        document = write(document)
        if checkpoint is not None:
            checkpoint(document)

    document = document.complete_stage(stage.name)
    if checkpoint is not None:
        checkpoint(document)
    return document


def discover_root_locators(document: Document, *, checkpoint: Checkpoint | None = None) -> Document:
    """Discover full reporter and docket occurrences in the source."""
    return _run_stage(document, _DISCOVERY, (find_full_reporter_locators, find_docket_locators), checkpoint)


def read_root_fields(
    document: Document,
    rules: ExtractionRules | None = None,
    *,
    checkpoint: Checkpoint | None = None,
) -> Document:
    """Read grounded full-citation fields using deterministic rules."""
    config = rules or stable()
    writers = (
        resolve_docket_entries,
        partial(resolve_colocations, rules=config),
        partial(resolve_case_names, rules=config),
        partial(resolve_courts, rules=config),
        partial(resolve_dates, rules=config),
        partial(resolve_pin_cites, rules=config),
    )
    return _run_stage(document, _READING, writers, checkpoint)


def form_root_groups(document: Document, *, checkpoint: Checkpoint | None = None) -> Document:
    """Assign each occurrence to its representative citation root."""
    return _run_stage(document, _GROUPING, (form_roots,), checkpoint)


def grow_roots(
    document: Document,
    *,
    rules: ExtractionRules | None = None,
    checkpoint: Checkpoint | None = None,
) -> Document:
    """Run the three root extraction stages or resume from a committed boundary."""
    _require_resume_prefix(document)
    if _DISCOVERY.name not in document.stage_runs:
        document = discover_root_locators(document, checkpoint=checkpoint)
    if _READING.name not in document.stage_runs:
        document = read_root_fields(document, rules, checkpoint=checkpoint)
    if _GROUPING.name not in document.stage_runs:
        document = form_root_groups(document, checkpoint=checkpoint)
    return document
