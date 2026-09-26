"""Render precision for reporter exact-lookup ambiguity decisions."""

from __future__ import annotations

from typing import Any

from evaluations.annotations import SETS


def _precision(metric: dict[str, Any]) -> str:
    value = metric.get("value")
    percent = f"{value:.1%}" if value is not None else "—"
    return f"{metric.get('correct', 0)}/{metric.get('scored', 0)} ({percent})"


def render_report(result: dict[str, Any], *, source_label: str) -> str:
    """Render admission and selected-candidate field precision with denominators."""
    if result.get("stage") != "reporter_root_exact_ambiguity" or not result.get("sets"):
        raise ValueError("Expected a nonempty reporter ambiguity-stage summary")
    lines = [
        f"# Precision at `{result['stage']}`",
        "",
        f"<!-- Generated from {source_label} by evaluations.render_reporter_ambiguity_report. -->",
        "",
        "Admission precision scores unique rule admissions against labeled root identity. Field precision "
        "scores the selected candidate's field judgments against explicit root-level field labels, "
        "including judgments on later occurrences of that root. Each value shows correct/scored "
        "judgments; a dash means no labeled judgment was scored.",
        "",
        "| Set | Admission | Case name | Court | Date |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in SETS:
        if name not in result["sets"]:
            continue
        lines.append(_row(name, result["sets"][name]))
    lines.append(_row("Total", result["totals"]))
    return "\n".join(lines) + "\n"


def _row(name: str, data: dict[str, Any]) -> str:
    fields = data.get("field_precision", {})
    return "| " + " | ".join((
        name,
        _precision(data.get("admission_precision", {})),
        *(_precision(fields.get(field, {})) for field in ("case_name", "court", "date")),
    )) + " |"
