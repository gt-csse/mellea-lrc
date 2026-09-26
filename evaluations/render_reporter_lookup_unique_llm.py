"""Render incremental field judgments for unique reporter model review."""

from __future__ import annotations

from typing import Any

from evaluations.annotations import SETS
from mellea_lrc.validation.reporter_root_lookup_unique_llm import STAGE

FIELDS = (("case_name", "Case name"), ("court", "Court"), ("date", "Date"))


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def render_report(result: dict[str, Any], *, source_label: str) -> str:
    if result.get("stage") != STAGE or not result.get("sets"):
        raise ValueError("Expected a nonempty unique-reporter model-stage summary")
    if set(result["sets"]) - set(SETS):
        raise ValueError("Unknown dataset in unique-reporter model-stage summary")
    lines = [
        f"# Field judgments at `{STAGE}`",
        "",
        f"<!-- Generated from {source_label} by evaluations.render_reporter_lookup_unique_llm. -->",
        "",
        "| Set | Field | Precision | Recall |",
        "| --- | --- | ---: | ---: |",
    ]
    for name, summary in (
        *((name, result["sets"][name]) for name in SETS if name in result["sets"]),
        ("Total", result["totals"]),
    ):
        for field, label in FIELDS:
            metric = summary["fields"][field]
            lines.append(
                f"| {name} | {label} | {_percent(metric['precision'])} | {_percent(metric['recall'])} |"
            )
    return "\n".join(lines) + "\n"
