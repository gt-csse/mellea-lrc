"""Render field judgment precision for ambiguous reporter model review."""

from __future__ import annotations

from typing import Any

from evaluations.annotations import SETS
from mellea_lrc.validation.stage_names import REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM as STAGE


def _precision(field: dict[str, Any]) -> str:
    value = field["value"]
    percent = f"{value:.1%}" if value is not None else "—"
    return f"{field['correct']}/{field['scored']} ({percent})"


def render_report(result: dict[str, Any], *, source_label: str) -> str:
    """Show only case-name, court, and date judgment precision."""
    if result.get("stage") != STAGE or not result.get("sets"):
        raise ValueError("Expected a nonempty ambiguous-reporter model-stage summary")
    if set(result["sets"]) - set(SETS):
        raise ValueError("Ambiguous-reporter model-stage summary includes an unknown set")
    lines = [
        f"# Field judgment precision at `{STAGE}`",
        "",
        f"<!-- Generated from {source_label} by evaluations.render_reporter_lookup_ambiguous_llm. -->",
        "",
        "Only this stage's selected-candidate case-name, court, and date judgments are counted. "
        "A judgment needs an explicit field label on the same annotated root occurrence, "
        "the latest aligned source reading, and its selected cluster in the canonical root's evidence. "
        "Values show correct/scored judgments; "
        "a dash means none were scored.",
        "",
        "| Set | Case name | Court | Date |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in SETS:
        if name in result["sets"]:
            lines.append(_row(name, result["sets"][name]))
    lines.append(_row("Total", result["totals"]))
    return "\n".join(lines) + "\n"


def _row(name: str, summary: dict[str, Any]) -> str:
    fields = summary["field_precision"]
    return f"| {name} | {_precision(fields['case_name'])} | {_precision(fields['court'])} | {_precision(fields['date'])} |"
