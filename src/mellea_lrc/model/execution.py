"""The immutable workflow hierarchy and durable checkpoint events."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True, slots=True)
class SubstageDefinition:
    """An atomic execution, commit, and recovery boundary."""

    name: str
    optional: bool = False


@dataclass(frozen=True, slots=True)
class StageDefinition:
    """An ordered group of substages with a meaningful task boundary."""

    name: str
    substages: tuple[SubstageDefinition, ...]
    optional: bool = False

    @property
    def required_substages(self) -> tuple[str, ...]:
        return tuple(substage.name for substage in self.substages if not substage.optional)


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    """An ordered collection of semantic stages."""

    name: str
    stages: tuple[StageDefinition, ...]

    @property
    def required_stages(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages if not stage.optional)


def _stage(
    workflow: str,
    name: str,
    substages: tuple[str, ...],
    *,
    optional_substages: tuple[str, ...] = (),
    optional: bool = False,
) -> StageDefinition:
    path = f"{workflow}.{name}"
    return StageDefinition(
        name=path,
        substages=tuple(
            SubstageDefinition(f"{path}.{substage}", substage in optional_substages) for substage in substages
        ),
        optional=optional,
    )


WORKFLOW_CATALOG: tuple[WorkflowDefinition, ...] = (
    WorkflowDefinition(
        "grow_roots",
        (
            _stage(
                "grow_roots",
                "locator_discovery",
                ("full_reporter_locators", "docket_locators", "docket_hunting"),
                optional_substages=("docket_hunting",),
            ),
            _stage(
                "grow_roots",
                "field_reading",
                ("docket_entries", "colocations", "case_names", "courts", "dates", "pin_cites"),
            ),
            _stage(
                "grow_roots",
                "root_formation",
                ("rule", "docket_llm_reassignment"),
                optional_substages=("docket_llm_reassignment",),
            ),
        ),
    ),
    WorkflowDefinition(
        "validate_roots",
        (
            _stage(
                "validate_roots",
                "reporter_lookup",
                (
                    "cluster_retrieval",
                    "docket_retrieval",
                    "unique_rule_judgment",
                    "ambiguous_rule_judgment",
                    "unique_llm_judgment",
                    "ambiguous_llm_judgment",
                ),
            ),
            _stage(
                "validate_roots",
                "docket_lookup",
                (
                    "courtlistener_retrieval",
                    "courtlistener_review",
                    "govinfo_retrieval",
                    "govinfo_review",
                    "identity_aggregation",
                ),
            ),
            _stage(
                "validate_roots",
                "locator_body_corroboration",
                (
                    "courtlistener_opinion_retrieval",
                    "courtlistener_recap_retrieval",
                    "govinfo_opinion_retrieval",
                    "llm_judgment",
                ),
            ),
            _stage(
                "validate_roots",
                "intended_case_discovery",
                (
                    "courtlistener_opinion_retrieval",
                    "courtlistener_recap_retrieval",
                    "govinfo_opinion_retrieval",
                    "llm_selection",
                ),
                optional=True,
            ),
        ),
    ),
    WorkflowDefinition(
        "grow_leaves",
        (
            _stage(
                "grow_leaves",
                "short_reporter_citations",
                ("discovery", "colocations", "case_names", "attribution"),
            ),
            _stage("grow_leaves", "reference_citations", ("discovery", "attribution")),
            _stage("grow_leaves", "id_citations", ("discovery", "attribution")),
            _stage(
                "grow_leaves",
                "supra_citations",
                ("discovery", "case_names", "pin_cites", "rule_attribution", "llm_attribution"),
                optional_substages=("llm_attribution",),
            ),
            _stage("grow_leaves", "leaf_field_correction", ("review",)),
        ),
    ),
    WorkflowDefinition(
        "validate_pincite",
        (
            _stage("validate_pincite", "opinion_preparation", ("retrieval", "page_index")),
            _stage(
                "validate_pincite",
                "citation_preparation",
                ("page_resolution", "opinion_review", "propositions", "evidence"),
            ),
            _stage("validate_pincite", "support_review", ("page_review", "full_opinion_review", "judgment")),
        ),
    ),
)

STAGE_CATALOG = MappingProxyType(
    {stage.name: stage for workflow in WORKFLOW_CATALOG for stage in workflow.stages}
)
SUBSTAGE_CATALOG = MappingProxyType(
    {substage.name: substage for stage in STAGE_CATALOG.values() for substage in stage.substages}
)
_SUBSTAGE_STAGES = MappingProxyType(
    {substage.name: stage.name for stage in STAGE_CATALOG.values() for substage in stage.substages}
)


def get_workflow(name: str) -> WorkflowDefinition:
    """Look up one workflow without importing execution implementations."""
    for workflow in WORKFLOW_CATALOG:
        if workflow.name == name:
            return workflow
    raise KeyError(f"Unknown workflow: {name}")


def get_stage_definition(name: str) -> StageDefinition:
    try:
        return STAGE_CATALOG[name]
    except KeyError as exc:
        raise KeyError(f"Unknown stage: {name}") from exc


def get_substage_definition(name: str) -> SubstageDefinition:
    try:
        return SUBSTAGE_CATALOG[name]
    except KeyError as exc:
        raise KeyError(f"Unknown substage: {name}") from exc


def stage_for_substage(name: str) -> str | None:
    """Return catalog membership; ad hoc development substages have no group."""
    return _SUBSTAGE_STAGES.get(name)


class CheckpointRun(BaseModel):
    """One append-only event; a stage marker has no citation decision node."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    kind: Literal["substage", "stage"]
    name: str = Field(min_length=1)
