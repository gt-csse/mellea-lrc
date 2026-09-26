"""Render only reporter exact-lookup field judgment precision and recall.

Run from the repository root::

    uv run python -m evaluations.render_identity_report \
        --summary evaluations/results/2026-09-25-reporter-exact-identity-primary-court/summary.json \
        --output local/reporter-exact-field-report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluations.annotations import SETS

FIELDS = (
    ("case_name", "Case name"),
    ("court", "Court"),
    ("date", "Date"),
)


def _percent(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "—"


def _field_row(name: str, label: str, field: dict[str, Any]) -> str:
    return f"| {name} | {label} | {_percent(field['precision'])} | {_percent(field['global_recall'])} |"


def render_identity_report(result: dict[str, Any], *, source_label: str) -> str:
    """Display the three field judgments using their saved scorer ratios."""
    if not result.get("stage") or not result.get("sets"):
        raise ValueError("Expected a nonempty identity-stage score summary")
    if set(result["sets"]) - set(SETS):
        raise ValueError("Identity-stage summary includes an unknown set")
    if result["totals"].get("documents") != sum(row.get("documents", 0) for row in result["sets"].values()):
        raise ValueError("Identity-stage document totals do not agree")
    lines = [
        f"# Field judgments at `{result['stage']}`",
        "",
        f"<!-- Generated from {source_label} by evaluations.render_identity_report. -->",
        "",
        "| Set | Field | Judgment precision | Judgment recall |",
        "| --- | --- | ---: | ---: |",
    ]
    for name in SETS:
        if name in result["sets"]:
            lines.extend(
                _field_row(name, label, result["sets"][name]["fields"][field]) for field, label in FIELDS
            )
    lines.extend(_field_row("Total", label, result["totals"]["fields"][field]) for field, label in FIELDS)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.summary.read_text(encoding="utf-8"))
    report = render_identity_report(result, source_label=str(args.summary))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
