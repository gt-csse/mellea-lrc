"""Render overall reporter-root field precision and recall."""

from __future__ import annotations

from typing import Any

from evaluations.score_reporter_fields_overall import FIELDS, NAME


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def render_report(result: dict[str, Any], *, source_label: str) -> str:
    """Show only the three requested field comparisons and their denominators."""
    if result.get("name") != NAME:
        raise ValueError("Expected the overall reporter-root field score")
    lines = [
        "# Overall reporter-root field judgments",
        "",
        f"<!-- Generated from {source_label} by evaluations.render_reporter_fields_overall. -->",
        "",
        "The checkpoint combines rule checks and both model-review routes. Precision is among "
        "decided judgments with a comparable annotated root, aligned source reading, and an "
        "evidence-linked selected record. Recall includes every explicitly labeled, unmasked "
        "canonical full-reporter root; unresolved roots count as misses. Masked canonical "
        "occurrences are excluded because a later citation to the same case can state different "
        "fields. A field marked not stated has no match/mismatch label and is excluded from "
        "that field's denominator.",
        "",
    ]
    for name, summary in result["sets"].items():
        population = summary["population"]
        lines.append(
            f"**{name}:** {population['reporter_locator_occurrences']} unmasked reporter locator "
            f"occurrences represent {population['reporter_identities']} distinct annotated "
            f"identities; {population['unmasked_canonical_roots']} have an unmasked canonical "
            "full-reporter root eligible for field recall."
        )
    lines.extend(
        (
            "",
            "| Set | Field | Full reporter locator roots | Precision | Recall |",
            "| --- | --- | ---: | ---: | ---: |",
        )
    )
    for name, summary in (*result["sets"].items(), ("Total", result["totals"])):
        for field in FIELDS:
            metric = summary["fields"][field]
            lines.append(
                f"| {name} | {field.replace('_', ' ')} | "
                f"{summary['population']['unmasked_canonical_roots']} | "
                f"{metric['correct']}/{metric['scored']} ({_percent(metric['precision'])}) | "
                f"{metric['correct']}/{metric['gold']} ({_percent(metric['recall'])}) |"
            )
    return "\n".join(lines) + "\n"
