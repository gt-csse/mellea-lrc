"""Additive score values shared by the independent workflow scorers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from mellea_lrc.model.execution import get_workflow


@dataclass(frozen=True)
class Precision:
    correct: int = 0
    total: int = 0

    def __add__(self, other: Precision) -> Precision:
        return Precision(self.correct + other.correct, self.total + other.total)

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "correct": self.correct,
            "total": self.total,
            "precision": self.correct / self.total if self.total else None,
        }


@dataclass(frozen=True)
class FieldScore:
    correct: int = 0
    predicted: int = 0
    gold: int = 0

    def __add__(self, other: FieldScore) -> FieldScore:
        return FieldScore(
            self.correct + other.correct, self.predicted + other.predicted, self.gold + other.gold
        )

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "correct": self.correct,
            "predicted": self.predicted,
            "gold": self.gold,
            "precision": self.correct / self.predicted if self.predicted else None,
            "recall": self.correct / self.gold if self.gold else None,
        }


@dataclass(frozen=True)
class SubstageScore:
    substage: str
    metrics: dict[str, Precision]

    def __add__(self, other: SubstageScore) -> SubstageScore:
        if self.substage != other.substage or self.metrics.keys() != other.metrics.keys():
            raise ValueError("Cannot combine different substage scores")
        return SubstageScore(
            self.substage, {key: value + other.metrics[key] for key, value in self.metrics.items()}
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "substage": self.substage,
            "metrics": {key: value.as_dict() for key, value in self.metrics.items()},
        }


def group_substage_records(
    workflow: str,
    records: Iterable[dict[str, Any]],
    completed_stages: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Group existing atomic records in catalog order without aggregating metrics."""
    by_name: dict[str, dict[str, Any]] = {}
    for record in records:
        name = record["substage"]
        if name in by_name:
            raise ValueError(f"Duplicate substage score: {name}")
        by_name[name] = record
    groups = []
    for stage in get_workflow(workflow).stages:
        members = [by_name.pop(substage.name) for substage in stage.substages if substage.name in by_name]
        if members:
            groups.append(
                {
                    "stage": stage.name,
                    "completed": stage.name in completed_stages,
                    "substages": members,
                }
            )
    if by_name:
        raise ValueError(f"Unknown {workflow} substages: {', '.join(by_name)}")
    return groups


def substage_heading(name: str) -> str:
    """Use local catalog numbering for each independently rendered substage."""
    workflow = get_workflow(name.split(".", 1)[0])
    for stage_index, stage in enumerate(workflow.stages, 1):
        for substage_index, substage in enumerate(stage.substages, 1):
            if substage.name == name:
                return f"### {stage_index}.{substage_index} {name}"
    raise ValueError(f"Unknown substage: {name}")


def render_stage_sections(
    workflow: str,
    sections: Iterable[tuple[str, str]],
    completed_stages: tuple[str, ...] = (),
) -> list[str]:
    """Nest each existing detail table under its semantic stage heading."""
    grouped = group_substage_records(
        workflow,
        ({"substage": name, "section": section} for name, section in sections),
        completed_stages,
    )
    indices = {stage.name: index for index, stage in enumerate(get_workflow(workflow).stages, 1)}
    rendered = []
    for group in grouped:
        partial = " (incomplete)" if not group["completed"] else ""
        rendered.append(f"## {indices[group['stage']]} {group['stage']}{partial}")
        rendered.extend(member["section"].rstrip() for member in group["substages"])
    return rendered
