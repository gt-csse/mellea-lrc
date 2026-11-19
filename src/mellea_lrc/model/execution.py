"""Durable completion events and the native root-extraction stage catalog."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CheckpointRun(BaseModel):
    """One completed substage or group of substages."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    kind: Literal["substage", "stage"] = "substage"
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){0,2}$")


@dataclass(frozen=True, slots=True)
class StageDefinition:
    """A named stage and its ordered atomic checkpoints."""

    name: str
    substages: tuple[str, ...]


def _grow_roots_stage(name: str, *substages: str) -> StageDefinition:
    name = f"grow_roots.{name}"
    return StageDefinition(name, tuple(f"{name}.{substage}" for substage in substages))


GROW_ROOTS_STAGES = (
    _grow_roots_stage("discover_root_locators", "find_full_reporter_locators", "find_docket_locators"),
    _grow_roots_stage(
        "read_root_fields",
        "resolve_docket_entries",
        "resolve_colocations",
        "resolve_case_names",
        "resolve_courts",
        "resolve_dates",
        "resolve_pin_cites",
    ),
    _grow_roots_stage("form_root_groups", "form_roots"),
)
GROW_ROOTS_RUNS = tuple(
    run
    for stage in GROW_ROOTS_STAGES
    for run in (
        *(CheckpointRun(name=substage) for substage in stage.substages),
        CheckpointRun(kind="stage", name=stage.name),
    )
)


def grow_roots_runs(runs: tuple[CheckpointRun, ...]) -> tuple[CheckpointRun, ...]:
    """Select the workflow's history without unrelated atomic checkpoints."""
    return tuple(run for run in runs if run.kind == "stage" or run.name.startswith("grow_roots."))
