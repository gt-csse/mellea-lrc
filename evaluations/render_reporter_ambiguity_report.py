"""Render ambiguity-stage admission precision/recall and route counts."""

from __future__ import annotations

from typing import Any

from evaluations.annotations import SETS


def _percent(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "—"


def render_report(result: dict[str, Any], *, source_label: str) -> str:
    """Render compact stage metrics and candidate route totals."""
    if result.get("stage") != "reporter_root_exact_ambiguity" or not result.get("sets"):
        raise ValueError("Expected a nonempty reporter ambiguity-stage summary")
    lines = [
        f"# Ambiguous reporter lookup decisions at `{result['stage']}`",
        "",
        f"<!-- Generated from {source_label} by evaluations.render_reporter_ambiguity_report. -->",
        "",
        "Admission means the rule stage selected one candidate as a unique match. "
        "Recall uses gold-correct reporter roots reached by an ambiguous exact lookup; "
        "candidate field judgments are retained as evidence, not scored against root-level field labels.",
        "",
        "| Set | Admission precision | Admission recall | Gold ambiguous roots | Rule admissions | Review required | Too many candidates |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in SETS:
        if name not in result["sets"]:
            continue
        row = result["sets"][name]
        routes = row.get("route_outcomes", {})
        lines.append(
            f"| {name} | {_percent(row.get('admission_precision'))} | {_percent(row.get('admission_recall'))} | "
            f"{row.get('gold_ambiguous_roots', 0)} | {routes.get('unique_rule_match', 0)} | "
            f"{routes.get('review_required', 0)} | {routes.get('too_many_candidates', 0)} |"
        )
    row = result["totals"]
    routes = row.get("route_outcomes", {})
    lines.append(
        f"| Total | {_percent(row.get('admission_precision'))} | {_percent(row.get('admission_recall'))} | "
        f"{row.get('gold_ambiguous_roots', 0)} | {routes.get('unique_rule_match', 0)} | "
        f"{routes.get('review_required', 0)} | {routes.get('too_many_candidates', 0)} |"
    )
    return "\n".join(lines) + "\n"
